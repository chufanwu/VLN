import Tkinter
import sys
import os

# add `/path/to/your/Matterport3DSimulator/build` to system path
sys.path.append('./build')
import MatterSim
import math
import cv2
import json
import argparse
import sys
import re
import string
import json
import time
import math
from collections import Counter
import numpy as np
import networkx as nx


def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3] - pose2['pose'][3]) ** 2 \
                + (pose1['pose'][7] - pose2['pose'][7]) ** 2 \
                + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j, conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

def discret_heading(heading):
    return round(heading / 30.0) * 30.0


class Annotation(object):

    def __init__(self, addr, instruction_id):
        self.WIDTH = 640
        self.HEIGHT = 640
        self.VFOV = math.radians(90)
        self.HFOV = self.VFOV * self.WIDTH / self.HEIGHT
        self.TEXT_COLOR = [230, 40, 40]
        self.DESTINATION_COLOR = [40, 40, 230]

        self.IMAGE_CACHE_ADDR = './img_cache.png'
        self.DATA_SAVE = '../AnnotationDataGoalMap/train/'
        if not os.path.exists(self.DATA_SAVE): os.mkdir(self.DATA_SAVE)
        with open(addr) as fp:
            train_data = json.load(fp)
        self.data = train_data[instruction_id]
        self.INSTRUCTION_ID = instruction_id
        self.SCAN_ID = self.data['scan']
        self.start_location = self.data['path'][0]
        self.start_heading = discret_heading(self.data['heading'])
        self.start_elevation = 0.0
        self.INSTRUCTION = ''
        for id, s in enumerate(self.data['instructions']):
            self.INSTRUCTION += str(id + 1) + '. ' + s
            self.INSTRUCTION += '\n\n'

        self.current_location = self.data['path'][0]
        self.current_heading = discret_heading(self.data['heading'])
        self.current_elevation = 0.0
        self.action_list = []
        self.is_annotating = False
        self.creat_GUI()

    def read_data(self):
        self.data_scan_id = self.data['scan']
        self.data_instruction = self.data['instructions'][0]
        self.start_location = self.data['path'][0]
        self.start_heading = self.data['heading']
        self.start_elevation = 0
        self.path = self.data['path']

        self.graph = load_nav_graphs([self.data_scan_id])
        self.paths = {}
        for scan, G in self.graph.iteritems():  # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graph.iteritems():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        degree30 = math.radians(30.0)
        if state.location.viewpointId == goalViewpointId:
            if state.viewIndex // 12 == 0:
                return (0, 0, degree30)
            elif state.viewIndex // 12 == 2:
                return (0, 0, -degree30)
            else:
                return (0, 0, 0)  # do nothing
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        # Can we see the next viewpoint?
        for i, loc in enumerate(state.navigableLocations):
            if loc.viewpointId == nextViewpointId:
                # Look directly at the viewpoint before moving
                if loc.rel_heading > (math.pi / 12.0):
                    return (0, degree30, 0)  # Turn right
                elif loc.rel_heading < -math.pi / 12.0:
                    return (0, -degree30, 0)  # Turn left
                elif loc.rel_elevation > math.pi / 12.0 and state.viewIndex // 12 < 2:
                    return (0, 0, degree30)  # Look up
                elif loc.rel_elevation < -math.pi / 12.0 and state.viewIndex // 12 > 0:
                    return (0, 0, -degree30)  # Look down
                else:
                    return (1, 0, 0)  # Move
        # Can't see it - first neutralize camera elevation
        if state.viewIndex // 12 == 0:
            return (0, 0, 1)  # Look up
        elif state.viewIndex // 12 == 2:
            return (0, 0, -1)  # Look down
        # Otherwise decide which way to turn
        target_rel = self.graph[state.scanId].node[nextViewpointId]['position'] - state.location.point
        target_heading = math.pi / 2.0 - math.atan2(target_rel[1], target_rel[0])  # convert to rel to y axis
        if target_heading < 0:
            target_heading += 2.0 * math.pi
        if state.heading > target_heading and state.heading - target_heading < math.pi:
            return (0, -1, 0)  # Turn left
        if target_heading > state.heading and target_heading - state.heading > math.pi:
            return (0, -1, 0)  # Turn left
        return (0, 1, 0)  # Turn right

    def creat_sim(self):
        self.read_data()
        self.sim = MatterSim.Simulator()
        self.sim.setCameraResolution(self.WIDTH, self.HEIGHT)
        self.sim.setCameraVFOV(self.VFOV)
        self.sim.setElevationLimits(-30 * math.pi / 180.0, 30 * math.pi / 180.0)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.init()
        self.sim.newEpisode(self.SCAN_ID, self.start_location, self.start_heading, self.start_elevation)
        self.sim_step(init=True)

    def sim_step(self, init=False):
        state = self.sim.getState()
        action = self._shortest_path_action(state, self.path[-1])
        location = action[0]
        heading = action[1]
        elevation = action[2]
        self.sim.makeAction(location, heading, elevation)
        state = self.sim.getState()
        locations = state.navigableLocations
        im = state.rgb
        if locations[1:]:
            for idx, loc in enumerate(locations[1:]):
                # Draw actions on the screen
                fontScale = 3.0 / loc.rel_distance
                x = int(self.WIDTH / 2 + loc.rel_heading / self.HFOV * self.WIDTH)
                y = int(self.HEIGHT / 2 - loc.rel_elevation / self.VFOV * self.HEIGHT)
                if idx == 1:
                    cv2.putText(im, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale, self.DESTINATION_COLOR, thickness=3)
                else:
                    cv2.putText(im, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale, self.TEXT_COLOR, thickness=3)
        cv2.imwrite(self.IMAGE_CACHE_ADDR, im)

        im = Tkinter.PhotoImage(file=self.IMAGE_CACHE_ADDR)
        self.view.configure(image=im)
        self.view.image = im
        self.current_location = state.location.viewpointId
        self.current_heading = state.heading
        self.current_elevation = state.elevation
        self.cur_location.configure(text='Current Location: %s' % self.current_location)
        self.cur_heading.configure(text='Current Heading: %s' % str(self.current_heading / math.pi * 180))
        self.cur_elevation.configure(text='Current Elevation: %s' % str(self.current_elevation / math.pi * 180))
        if not init:

            s = self.sim.getState()
            viewId = int(1 + ((self.current_elevation / math.pi * 180.0) / 30.0)) * 12 + int((self.current_heading / math.pi * 180.0) / 30.0)
            imageId = self.current_location + '_%d' % viewId
            neibor = s.navigableLocations
            neiborInfo = []
            for n in neibor:
                neiborInfo.append({'viewpointId': n.viewpointId,
                                   'rel_heading': n.rel_heading,
                                   'rel_elevation': n.rel_elevation,
                                   'rel_distance': n.rel_distance,
                                   'point': n.point})

            action_to_record = {'type': 'action',
                                'location': location,
                                'heading': heading,
                                'elevation': elevation}
            state_to_record = {'type': 'state',
                               'location': self.current_location,
                               'heading': self.current_heading,
                               'elevation': self.current_elevation,
                               'point' : s.location.point,
                               'imageId': imageId,
                               'neibors' : neiborInfo}

            self.action_list.append(action_to_record)
            self.action_list.append(state_to_record)
            if action[0] != 0:
                s = 'Forward'
                self.record.insert(Tkinter.END, s)
            elif action[1] != 0:
                s = 'Change heading by: %.3f degree' % (action[1] * 30)
                self.record.insert(Tkinter.END, s)
            elif action[2] != 0:
                s = 'Change elevation by: %.3f degree' % (action[2] * 30)
                self.record.insert(Tkinter.END, s)

    def save_and_start_new(self):

        # path = self.DATA_SAVE + self.SCAN_ID
        # if not os.path.exists(path):
        #     os.mkdir(path)

        self.record.delete(0, Tkinter.END)

        commands = self.action_list[0]['instruction'].split(';')
        for c in commands:
            self.action_list[0]['instruction'] = c
            index = 1
            path = os.path.join(self.DATA_SAVE, '%d' % self.INSTRUCTION_ID)
            while os.path.exists(path + '_%d.json' % index):
                index += 1
            path = path + '_%d.json' % index
            with open(path, 'w') as fp:
                json.dump(self.action_list, fp, indent=4)
            self.record.insert(Tkinter.END, 'Annotation data saved to %s' % path)


        self.annotating.configure(text='Not Annotating')
        self.is_annotating = False

        # start a new one
        self.start_location = self.current_location
        self.start_heading = self.current_heading
        self.start_elevation = self.current_elevation
        self.intro_start_location.configure(text='Start Location: %s    ' % self.start_location)
        self.intro_start_heading.configure(text='Start Heading: %s  ' % str(self.start_heading / math.pi * 180))
        self.intro_start_elevation.configure(text='Start Elevation: %s  ' % str(self.start_elevation / math.pi * 180))
        self.action_list = []


    def confirm_command_and_start(self):
        command = self.command.get()
        self.annotating.configure(text='Annotating:  %s' % command)
        self.action_list.append({'type': 'title',
                                 'scan_id': self.SCAN_ID,
                                 'instruction': command,
                                 'instructionID': self.INSTRUCTION_ID})


        s = self.sim.getState()

        viewId = int(1 + ((self.current_elevation / math.pi * 180.0) / 30.0)) * 12 + int(
            (self.current_heading / math.pi * 180.0) / 30.0)
        imageId = self.current_location + '_%d' % viewId

        neibor = s.navigableLocations
        neiborInfo = []
        for n in neibor:
            neiborInfo.append({'viewpointId' : n.viewpointId,
                               'rel_heading' : n.rel_heading,
                               'rel_elevation' : n.rel_elevation,
                               'rel_distance' : n.rel_distance,
                               'point' : n.point})


        self.action_list.append({'type': 'state',
                                 'location': self.current_location,
                                 'heading': self.current_heading,
                                 'elevation': self.current_elevation,
                                 'point': s.location.point,
                                 'imageId' : imageId,
                                 'neibors' : neiborInfo})

        self.record.delete(0, Tkinter.END)
        self.short_command.delete(0, Tkinter.END)
        self.record.insert(Tkinter.END, json.dumps({'type': 'title',
                                                    'scan_id': self.SCAN_ID,
                                                    'instruction': command}))
        self.record.insert(Tkinter.END, json.dumps({'type': 'state',
                                                    'location': self.current_location,
                                                    'heading': self.current_heading,
                                                    'elevation': self.current_elevation}))
        self.is_annotating = True

    def reset_current_annotation(self):
        self.sim.newEpisode(self.SCAN_ID, self.start_location, self.start_heading, self.start_elevation)
        self.record.delete(0, Tkinter.END)
        self.record.insert(Tkinter.END, 'Annotation Reset')
        self.short_command.delete(0, Tkinter.END)
        self.annotating.configure(text='Not Annotating')
        self.is_annotating = False

        # start a new one
        self.intro_start_location.configure(text='Start Location: %s    ' % self.start_location)
        self.intro_start_heading.configure(text='Start Heading: %s  ' % str(self.start_heading / math.pi * 180))
        self.intro_start_elevation.configure(text='Start Elevation: %s  ' % str(self.start_elevation / math.pi * 180))
        self.current_location = self.start_location
        self.current_heading = self.start_heading
        self.current_elevation = self.start_elevation
        self.cur_location.configure(text='Current Location: %s' % self.current_location)
        self.cur_heading.configure(text='Current Heading: %s' % str(self.current_heading / math.pi * 180))
        self.cur_elevation.configure(text='Current Elevation: %s' % str(self.current_elevation / math.pi * 180))
        self.action_list = []
        self.sim_step(init=True)

    def reset_entire_annotation(self):

        self.SCAN_ID = self.data['scan']
        self.start_location = self.data['path'][0]
        self.start_heading = self.data['heading']
        self.start_elevation = 0.0
        self.current_location = self.start_location
        self.current_heading = self.start_heading
        self.current_elevation = self.start_elevation

        self.sim.newEpisode(self.SCAN_ID, self.start_location, self.start_heading, self.start_elevation)
        self.record.delete(0, Tkinter.END)
        self.record.insert(Tkinter.END, 'Annotation Reset')
        self.short_command.delete(0, Tkinter.END)
        self.annotating.configure(text='Not Annotating')
        self.is_annotating = False

        # start a new one
        self.intro_start_location.configure(text='Start Location: %s    ' % self.start_location)
        self.intro_start_heading.configure(text='Start Heading: %s  ' % str(self.start_heading / math.pi * 180))
        self.intro_start_elevation.configure(text='Start Elevation: %s  ' % str(self.start_elevation / math.pi * 180))
        self.cur_location.configure(text='Current Location: %s' % self.current_location)
        self.cur_heading.configure(text='Current Heading: %s' % str(self.current_heading / math.pi * 180))
        self.cur_elevation.configure(text='Current Elevation: %s' % str(self.current_elevation / math.pi * 180))
        self.action_list = []
        self.sim_step(init=True)

    def set_current_as_start(self):
        self.is_annotating = False
        self.record.delete(0, Tkinter.END)
        self.record.insert(Tkinter.END, 'Set current location as the start point. Please enter the command.')
        self.short_command.delete(0, Tkinter.END)
        self.annotating.configure(text='Not Annotating')

        self.start_location = self.current_location
        self.start_heading = self.current_heading
        self.start_elevation = self.current_elevation
        self.intro_start_location.configure(text='Start Location: %s    ' % self.start_location)
        self.intro_start_heading.configure(text='Start Heading: %s  ' % str(self.start_heading / math.pi * 180))
        self.intro_start_elevation.configure(text='Start Elevation: %s  ' % str(self.start_elevation / math.pi * 180))
        self.action_list = []

    def creat_GUI(self):
        self.win = Tkinter.Tk()
        self.frame = Tkinter.Frame(self.win)

        self.intro_scan_id = Tkinter.Label(self.frame, font='Helvetica -20 normal',
                                           text='Scan ID: %s    ' % self.SCAN_ID)
        self.intro_instruction_id = Tkinter.Label(self.frame, font='Helvetica -20 normal',
                                                  text='Instruction ID: %s    ' % self.INSTRUCTION_ID)
        self.intro_start_location = Tkinter.Label(self.frame, font='Helvetica -20 normal',
                                                  text='Start Location: %s       ' % self.start_location)
        self.intro_start_heading = Tkinter.Label(self.frame, font='Helvetica -20 normal',
                                                 text='Start Heading: %f         ' % (
                                                         self.start_heading / math.pi * 180))
        self.intro_start_elevation = Tkinter.Label(self.frame, font='Helvetica -20 normal',
                                                   text='Start Elevation: %f         ' % (
                                                           self.start_elevation / math.pi * 180))
        self.cur_location = Tkinter.Label(self.frame, font='Helvetica -20 normal')
        self.cur_heading = Tkinter.Label(self.frame, font='Helvetica -20 normal')
        self.cur_elevation = Tkinter.Label(self.frame, font='Helvetica -20 normal')
        self.annotating = Tkinter.Label(self.frame, font='Helvetica -20 normal', text='Not Annotating')

        self.intro_scan_id.grid(row=1, column=1, sticky=Tkinter.W)
        self.intro_instruction_id.grid(row=1, column=2, sticky=Tkinter.W)
        self.intro_start_location.grid(row=2, column=1, sticky=Tkinter.W)
        self.intro_start_heading.grid(row=3, column=1, sticky=Tkinter.W)
        self.intro_start_elevation.grid(row=4, column=1, sticky=Tkinter.W)
        self.cur_location.grid(row=2, column=2, sticky=Tkinter.W)
        self.cur_heading.grid(row=3, column=2, sticky=Tkinter.W)
        self.cur_elevation.grid(row=4, column=2, sticky=Tkinter.W)
        self.annotating.grid(row=1, column=3)

        self.instruction = Tkinter.Text(self.frame, font='Helvetica -20 normal', width=91, height=10)
        self.instruction.insert(Tkinter.END, 'Instructions: \n\n%s' % self.INSTRUCTION)
        self.instruction.grid(row=5, column=1, columnspan=2, sticky=Tkinter.W)

        Tkinter.Label(self.frame, text='Command for current annotation: ', font='Helvetica -20 normal').grid(row=6,
                                                                                                             column=1,
                                                                                                             sticky=Tkinter.W)
        self.command = Tkinter.StringVar()
        self.short_command = Tkinter.Entry(self.frame, width=50, textvariable=self.command)
        self.short_command.grid(row=6, column=2, sticky=Tkinter.W)

        self.view = Tkinter.Label(self.frame, width=640, height=640)
        self.view.grid(row=2, column=3, rowspan=5)

        self.record = Tkinter.Listbox(self.frame, width=80)
        self.record.grid(row=6, column=3, rowspan=7)

        # buttons
        button_confirm_command = Tkinter.Button(self.frame, text='Confirm Command and Start Annotation',
                                                command=self.confirm_command_and_start, font='Helvetica -20 normal',
                                                width=40)
        button_confirm_command.grid(row=7, column=1, columnspan=2)

        button_save_and_start_new = Tkinter.Button(self.frame, text='Save and Start New',
                                                   command=self.save_and_start_new, font='Helvetica -20 normal',
                                                   width=40)
        button_save_and_start_new.grid(row=8, column=1, columnspan=2)

        button_set_current_as_start = Tkinter.Button(self.frame, text='Set Current Location as Start Point',
                                                     command=self.set_current_as_start, font='Helvetica -20 normal',
                                                     width=40)
        button_set_current_as_start.grid(row=9, column=1, columnspan=2)

        button_reset_current = Tkinter.Button(self.frame, text='Reset Current Annotation',
                                              command=self.reset_current_annotation, font='Helvetica -20 normal',
                                              width=40)
        button_reset_current.grid(row=10, column=1, columnspan=2)

        button_reset_entire = Tkinter.Button(self.frame, text='Reset Entire Annotation',
                                             command=self.reset_entire_annotation, font='Helvetica -20 normal',
                                             width=40)
        button_reset_entire.grid(row=11, column=1, columnspan=2)

        # pack all
        self.frame.pack()
        self.win.bind('<Key>', self.read_key)

        self.creat_sim()

        self.win.mainloop()

    def read_key(self, event):
        if not self.is_annotating:
            return
        self.short_command.delete(0, Tkinter.END)
        if event.char == 'f':
            self.sim_step()
        # elif event.char == 's':
        #     self.save_and_start_new()
        # elif event.char == 'r':
        #     self.reset_current_annotation()
        else:
            return

        # if event.char == 'w':
        #     elevation = self.ANGLEDELTA
        # elif event.char == 's':
        #     elevation = -self.ANGLEDELTA
        # elif event.char == 'a':
        #     heading = -self.ANGLEDELTA
        # elif event.char == 'd':
        #     heading = self.ANGLEDELTA
        # elif event.char == 'f':
        #     locations_sorted = locations[1:]
        #     locations_sorted.sort(key=lambda x: x.rel_distance)
        #     nearest_idx = locations[1:].index(locations_sorted[0])
        #     location = nearest_idx + 1
        # else:
        #     return
        #
        # self.sim_step(location, heading, elevation)


class View_Annotation(object):
    def __init__(self, data_addr):
        self.WIDTH = 640
        self.HEIGHT = 480
        self.VFOV = math.radians(60)
        self.HFOV = self.VFOV * self.WIDTH / self.HEIGHT
        self.TEXT_COLOR = [230, 40, 40]
        self.ANGLEDELTA = 15 * math.pi / 180
        self.DATA_ADDR = data_addr
        self.IMAGE_CACHE_ADDR = './img_cache.png'
        self.creat_GUI()

    def read_data(self):
        with open(self.DATA_ADDR, 'r') as fp:
            self.data_ori = json.load(fp)
        self.data_scan_id = self.data_ori[0]['scan_id']
        self.data_instruction = self.data_ori[0]['instruction']
        self.start_location = self.data_ori[1]['location']
        self.start_heading = self.data_ori[1]['heading']
        self.start_elevation = self.data_ori[1]['elevation']
        self.data_action = [(i['location'], i['heading'], i['elevation']) for i in self.data_ori if
                            i['type'] == 'action']

    def creat_sim(self):
        self.sim = MatterSim.Simulator()
        self.sim.setCameraResolution(self.WIDTH, self.HEIGHT)
        self.sim.setCameraVFOV(self.VFOV)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.init()
        self.sim.newEpisode(self.data_scan_id, self.start_location, self.start_heading, self.start_elevation)
        self.sim_step(0, 0, 0, init=True)

    def sim_step(self, location, heading, elevation, init=False):
        state = self.sim.getState()
        locations = state.navigableLocations
        if not locations[1:]:
            location = 0
        # nearest_idx = 0
        # if locations[1:] and location:
        #     locations_sorted = locations[1:]
        #     locations_sorted.sort(key=lambda x: x.rel_distance)
        #     nearest_idx = locations[1:].index(locations_sorted[0]) + 1

        self.sim.makeAction(location, heading, elevation)
        state = self.sim.getState()
        locations = state.navigableLocations
        im = state.rgb
        for idx, loc in enumerate(locations[1:]):
            # Draw actions on the screen
            fontScale = 3.0 / loc.rel_distance
            x = int(self.WIDTH / 2 + loc.rel_heading / self.HFOV * self.WIDTH)
            y = int(self.HEIGHT / 2 - loc.rel_elevation / self.VFOV * self.HEIGHT)
            cv2.putText(im, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, self.TEXT_COLOR, thickness=3)
        cv2.imwrite(self.IMAGE_CACHE_ADDR, im)

        im = Tkinter.PhotoImage(file=self.IMAGE_CACHE_ADDR)
        self.view.configure(image=im)
        self.view.image = im
        self.current_location = state.location.viewpointId
        self.current_heading = state.heading
        self.current_elevation = state.elevation
        self.cur_location.configure(text='Current Location: %s' % self.current_location)
        self.cur_heading.configure(text='Current Heading: %s' % str(self.current_heading / math.pi * 180))
        self.cur_elevation.configure(text='Current Elevation: %s' % str(self.current_elevation / math.pi * 180))

        if not init:
            if location != 0:
                s = 'To location: %d' % location
            elif heading != 0:
                s = 'Change heading by: %.3f' % heading
            else:
                s = 'Change elevation by: %.3f' % elevation
            self.record.insert(Tkinter.END, s)

    def change_image(self):
        if not self.data_action:
            del self.sim
            self.win.destroy()
            return

        current_action = self.data_action.pop(0)
        self.sim_step(current_action[0], current_action[1], current_action[2])
        self.view.after(300, self.change_image)

    def creat_GUI(self):
        self.read_data()
        self.win = Tkinter.Tk()
        self.frame = Tkinter.Frame(self.win)

        self.intro_scan_id = Tkinter.Label(self.frame, font='Helvetica -20 normal',
                                           text='Scan ID: %s' % self.data_scan_id)
        self.intro_start_location = Tkinter.Label(self.frame, font='Helvetica -20 normal',
                                                  text='Start Location: %s' % self.start_location)
        self.intro_start_heading = Tkinter.Label(self.frame, font='Helvetica -20 normal',
                                                 text='Start Heading: %f' % (
                                                         self.start_heading / math.pi * 180))
        self.intro_start_elevation = Tkinter.Label(self.frame, font='Helvetica -20 normal',
                                                   text='Start Elevation: %f' % (
                                                           self.start_elevation / math.pi * 180))
        self.cur_location = Tkinter.Label(self.frame, font='Helvetica -20 normal')
        self.cur_heading = Tkinter.Label(self.frame, font='Helvetica -20 normal')
        self.cur_elevation = Tkinter.Label(self.frame, font='Helvetica -20 normal')
        self.instruction = Tkinter.Label(self.frame, font='Helvetica -30 normal', text=self.data_instruction)

        self.intro_scan_id.grid(row=1, column=1, sticky=Tkinter.W)
        self.intro_start_location.grid(row=3, column=1, sticky=Tkinter.W)
        self.intro_start_heading.grid(row=4, column=1, sticky=Tkinter.W)
        self.intro_start_elevation.grid(row=5, column=1, sticky=Tkinter.W)
        self.cur_location.grid(row=6, column=1, sticky=Tkinter.W)
        self.cur_heading.grid(row=7, column=1, sticky=Tkinter.W)
        self.cur_elevation.grid(row=8, column=1, sticky=Tkinter.W)

        self.instruction.grid(row=1, column=2, rowspan=2)

        self.view = Tkinter.Label(self.frame, width=640, height=480)
        self.view.after(1000, self.change_image)
        self.view.grid(row=3, column=2, rowspan=10)

        self.record = Tkinter.Listbox(self.frame, width=80)
        self.record.grid(row=9, column=1, rowspan=4)

        # pack all
        self.frame.pack()

        self.creat_sim()

        self.win.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', required=True)
    parser.add_argument('--id')
    parser.add_argument('--scan')
    args = parser.parse_args()

    if args.func == 'a':
        addr = './tasks/R2R/data/R2R_train_for_45scans.json'
        anno = Annotation(addr, int(args.id))
    elif args.func == 'v':
        addr = './annotation_data_new/' + args.scan + '/' + args.scan + '_' + '0' * (
                    3 - len(args.id)) + args.id + '.json'
        View_Annotation(addr)
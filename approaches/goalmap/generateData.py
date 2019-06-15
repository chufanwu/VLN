import os
import json
import PIL.Image as Image
import matplotlib.pyplot as plt
import cv2
import numpy as np


def rotateMatrix(ang, tx, ty):
    # ang in rad
    ang = -ang
    M = np.array([np.cos(ang), -np.sin(ang), tx,
                  np.sin(ang), np.cos(ang), ty,
                  0, 0, 1])
    M = M.reshape(3, 3)
    M_i = np.linalg.inv(M)

    return M_i


def LocationSign(x, z, radius=1.0, dense=100, signColor=(255, 0, 0)):
    xx = np.linspace(x - radius, x + radius, dense).astype('float32')
    xx = xx.repeat(dense).reshape(-1, 1)

    zz = np.linspace(z - radius, z + radius, dense).astype('float32')
    zz = np.tile(zz, [dense]).reshape(-1, 1)

    yy = np.zeros(dense * dense).reshape(-1, 1).astype('float32')

    flag = ((np.power(xx - x, 2) + np.power(zz - z, 2)) <= np.power(radius, 2)).reshape(-1)

    xx = xx[flag, :]
    yy = yy[flag, :]
    zz = zz[flag, :]

    coor = np.concatenate((xx, yy, zz), axis=1)

    color = np.ones((coor.shape[0], 3)).astype('uint8')
    color[:, 0] = signColor[0]
    color[:, 1] = signColor[1]
    color[:, 2] = signColor[2]

    return coor, color


def TopView(rgbImage, depthImage, locs=None, imgScale=1.0, topViewSize=1000, height_scale=1.0):
    # Resize images
    rgbImage = cv2.resize(rgbImage, (0, 0), fx=imgScale, fy=imgScale)
    depthImage = cv2.resize(depthImage, (0, 0), fx=imgScale, fy=imgScale)
    (H, W, cc) = rgbImage.shape
    imgScale = H / 640.0

    mask = (depthImage >= (2.5 * depthImage.mean()))
    depthImage[mask] = 0.1

    # Calculate 3D coordinates
    CAMERA_FACTOR = 4000.0

    cx = 320.0 * imgScale
    cy = 320.0 * imgScale
    fx = 320.0 * imgScale
    fy = 320.0 * imgScale

    pz = depthImage * 65535 / CAMERA_FACTOR  # pz of size  H * W * 1
    mesh_px = np.repeat(np.array(range(0, W)).reshape(1, -1), H, axis=0).astype('float32')
    px = (mesh_px - cx) * pz / fx

    mesh_py = range(0, H)
    mesh_py.reverse()
    mesh_py = np.repeat(np.array(mesh_py).reshape(-1, 1), W, axis=1).astype('float32')
    py = (mesh_py - cy) * pz / fy
    # pz = -1 * pz

    px = px.reshape(H, W, 1)
    py = py.reshape(H, W, 1)
    pz = pz.reshape(H, W, 1)

    coor = np.concatenate((px, py, pz), axis=2).astype('float32')
    coor = coor.reshape(-1, 3)
    color = rgbImage.reshape(-1, 3)

    oriCoorLen = coor.shape[0]
    # add ego-location sign to topview
    if locs is not None:
        LocCoor, LocColor = LocationSign(0, 0, radius=0.5, signColor=(0, 255, 0))
        coor = np.concatenate((coor, LocCoor), axis=0)
        color = np.concatenate((color, LocColor), axis=0)

        while len(locs) > 1:
            loc = locs.pop(0)
            LocCoor, LocColor = LocationSign(loc[0], loc[1], radius=0.5, signColor=(127, 0, 0))
            coor = np.concatenate((coor, LocCoor), axis=0)
            color = np.concatenate((color, LocColor), axis=0)

        loc = locs.pop(0)
        LocCoor, LocColor = LocationSign(loc[0], loc[1], radius=0.5, signColor=(255, 0, 0))
        coor = np.concatenate((coor, LocCoor), axis=0)
        color = np.concatenate((color, LocColor), axis=0)

    # put the real world coordinates to pixels in the top view image
    x_min = np.min(coor[:, 0])
    y_min = np.min(coor[:, 1])
    z_min = np.min(coor[:, 2])

    coor[:, 0] -= x_min
    coor[:, 1] -= y_min
    coor[:, 2] -= z_min

    x_max = np.max(coor[:, 0])
    y_max = np.max(coor[:, 1])
    z_max = np.max(coor[:, 2])

    if x_max > z_max:
        zoom_in = (topViewSize - 2) / x_max
    else:
        zoom_in = (topViewSize - 2) / z_max

    coor[:, 0] *= zoom_in
    coor[:, 1] *= zoom_in
    coor[:, 2] *= zoom_in

    coor = np.ceil(coor).astype('int32')

    # x_size = np.ceil(x_max * zoom_in).astype('int32')
    # y_size = np.ceil(y_max * zoom_in).astype('int32')
    z_size = np.ceil(z_max * zoom_in).astype('int32')

    # HEIGHT_TRESHOLD = y_size * height_scale
    top_view_color = np.zeros((topViewSize, topViewSize, 3)).astype('uint8')
    top_view_goalmap = np.zeros((topViewSize, topViewSize)).astype('uint8')
    ############################# FAST METHOD ################################
    x = coor[:, 0]
    # y = coor[:, 1]
    z = coor[:, 2]

    # mask = (y <= HEIGHT_TRESHOLD).reshape(-1)
    # z = z[mask]
    # x = x[mask]

    z = topViewSize - 2 - z
    # top_view_color[z, x, :] = color[mask, :]
    top_view_color[z, x, :] = color[:, :]

    # for goal map
    coor = coor[oriCoorLen:, :]
    color = color[oriCoorLen:, :]
    x = coor[:, 0]
    # y = coor[:, 1]
    z = coor[:, 2]

    # mask = (y <= HEIGHT_TRESHOLD).reshape(-1)
    # z = z[mask]
    # x = x[mask]

    z = topViewSize - 2 - z
    # top_view_color[z, x, :] = color[mask, :]

    top_view_goalmap[z, x] = color[:, 0]
    return top_view_color, top_view_goalmap


def generateImage(dataPath):
    with open(dataPath, 'r') as fp:
        data = json.load(fp)

    dataTitle = data[0]
    scan = dataTitle['scan_id']
    instruction = dataTitle['instruction']
    dataState = [i for i in data if i['type'] == 'state']

    locIdSeq = []
    locImageIdSeq = []
    locPointSeq = []
    locHeadingSeq = []

    for i in dataState:
        locIdSeq.append(i['location'])
        locImageIdSeq.append(i['imageId'])
        locPointSeq.append(i['point'])
        locHeadingSeq.append(i['heading'])

    while len(locIdSeq) > 1:
        loc = locIdSeq[0]
        locImageId = locImageIdSeq[0]
        locRgbImage = plt.imread(os.path.join(rgbRootPath, scan, locImageId + '.jpg'))
        locDepthImage = plt.imread(os.path.join(depthRootPath, scan, locImageId + '.png'))

        resizeScale = 1
        # resizeScale = 28.0 / 640.0
        locRgbImage = cv2.resize(locRgbImage, (0, 0), fx=resizeScale, fy=resizeScale)
        locDepthImage = cv2.resize(locDepthImage, (0, 0), fx=resizeScale, fy=resizeScale)

        locPoint = locPointSeq[0]
        rotateM = rotateMatrix(locHeadingSeq[0], locPoint[0], locPoint[1])
        indicates = []

        # for nextId in range(1,len(locIdSeq)):
        for nextId in range(len(locIdSeq) - 1, len(locIdSeq)):
            nextPoint = np.array(locPointSeq[nextId][0:2] + [1.0]).reshape(3, 1)
            xyz = rotateM.dot(nextPoint).reshape(-1)
            dx, dy = xyz[0], xyz[1]
            indicates.append([dx, dy])

        topViewSize = 224
        topViewColor, topViewGoalmap = TopView(locRgbImage, locDepthImage, topViewSize=topViewSize, locs=indicates)

        plt.switch_backend('agg')
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(locRgbImage)
        plt.subplot(1, 3, 2)
        plt.imshow(topViewColor)
        plt.subplot(1, 3, 3)
        plt.imshow(topViewGoalmap)
        plt.suptitle(instruction, fontsize=16)
        plt.ioff()
        # Save the figures

        dataId = dataPath.split('/')[-1]
        dataId = dataId.split('.')[0]
        idx = 1
        colorSavePath = os.path.join(imageColorSaveRootPath, dataId + '_%d_color.png' % idx)
        goalmapSavePath = os.path.join(imageDataSaveRootPath, dataId + '_%d.png' % idx)

        while os.path.exists(colorSavePath):
            idx += 1
            colorSavePath = os.path.join(imageColorSaveRootPath, dataId + '_%d_color.png' % idx)
            goalmapSavePath = os.path.join(imageDataSaveRootPath, dataId + '_%d.png' % idx)

        plt.savefig(colorSavePath)
        cv2.imwrite(goalmapSavePath, topViewGoalmap)

        # Record the data information
        d = {'scan': scan,
             'instruction': instruction,
             'imageId': locImageId,
             'goalmapId': goalmapSavePath.split('/')[-1]
             }
        dataRecord.append(d)

        locIdSeq.pop(0)
        locImageIdSeq.pop(0)
        locPointSeq.pop(0)
        locHeadingSeq.pop(0)


ImageRootPath = '/Volumes/Dongqiyuan/matterport_dataset'
rgbRootPath = os.path.join(ImageRootPath, 'sampled_color_images')
depthRootPath = os.path.join(ImageRootPath, 'sampled_depth_images')
dataRootPath = '/Users/dongqiyuan/Desktop/GoalMap/AnnotationDataGoalMap/goalmap_training_raw_data'
imageDataSaveRootPath = '/Users/dongqiyuan/Desktop/GoalMap/AnnotationDataGoalMap/goalmap_training_image'
imageColorSaveRootPath = '/Users/dongqiyuan/Desktop/GoalMap/AnnotationDataGoalMap/goalmap_training_color_data'
jsonDataSavePath = '/Users/dongqiyuan/Desktop/GoalMap/AnnotationDataGoalMap'

datas = os.listdir(dataRootPath)
dataRecord = []

for idx, data in enumerate(datas):
    dataPath = os.path.join(dataRootPath, data)
    generateImage(dataPath)

    if idx % 10 == 0:
        print 'Finish %d / %d (%.1f%%)' % (idx + 1, len(datas), ((float(idx) + 1.0) / len(datas)) * 100.0)
        with open(os.path.join(jsonDataSavePath, 'training_data.json'), 'w') as fp:
            json.dump(dataRecord, fp, indent=4)

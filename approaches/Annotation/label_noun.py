import os, os.path
import json
from noun_extract import Noun_extractor
import ast
PATH = '/home/chufanw/Annotation/Annotation/'
PATH_new = os.path.join(PATH,'out')

def extract_noun(jname, rmlist):
    #extract the nouns from a speicific json file and delete the item from remove list
    with open(os.path.join(PATH, 'train', jname),"r") as jfile:
        nouns = set()
        data = json.load(jfile)
        instruc = data[0]['instruction']
        if instruc =='':
            print "Empty instruction"
            return 0
        else:
            extractor = Noun_extractor()
            print instruc
            nouns = extractor.get_noun(instruc)
            for item in rmlist:
                if item in nouns:
                    nouns.remove(item)
            data[0]['nouns']=list(nouns)
            return data

def get_rmlist():
    #get the list to remove some elements
    with open(os.path.join(PATH, "rmlist.txt"),"r") as rmfile:
        content = rmfile.readlines()
        content = [x.strip() for x in content]
        return content

def get_json():
    #get the file for all json files
    with open(os.path.join(PATH, "all_json.txt"),"r") as jfile:
        content = jfile.readlines()
        content = [x.strip() for x in content]
        return content


def dump_json(data, jname):
    with open(os.path.join(PATH, 'out',jname),'w') as jfile:
        print "hello"

def string2noun(rline):
    rline = rline[rline.find('['):rline.find(']') + 1]
    out = ast.literal_eval(rline)
    out = [x.strip() for x in out]
    return out


if __name__ == '__main__':
    rmlist = get_rmlist()
    jlist = get_json()
    #data = extract_noun(jlist[0],rmlist)
    with open(PATH+'log.txt',"r") as logfile:
        content = logfile.readlines()
        count = 0
        for eachj in jlist:
            data = extract_noun(eachj,rmlist)
            if data == 0:
                continue
            else:
                list_nouns = string2noun(content[count])
                data[0]['nouns'] = list_nouns
                with open(os.path.join(PATH_new, eachj),'w') as jfile:
                    json.dumnp(data,jfile,intent = 4)
                count += 1

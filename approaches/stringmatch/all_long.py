#get all long instructions
import json
def file2str(addr):
    with open(addr,"r") as file:
        content = file.readlines()
        content = [x.strip() for x in content]
        return content

def str2file(addr, list):
    with open(addr, "w") as file:
        for item in list:
            file.write("%s\n"%item)

def read_json(filename):
    with open(filename, "r") as jfile:
        content = json.load(jfile)
        return content

if __name__ == '__main__':
    all_long = []
    json_file = read_json('R2R_train.json')
    for item in json_file:
        for i in range(3):
            if len(item['instructions']) > 1:
                all_long.append(item['instructions'][i].lower())
                #all_long.append("I am wcf.")
    str2file("all_long.txt",all_long)

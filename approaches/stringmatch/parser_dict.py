# gererating json file
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

if __name__ == '__main__':
    json_list = []
    my_dict = {}
    file_content = file2str("parser.txt")
    count = -1
    for line in file_content:
        count += 1
        if count %2 == 0:
            tmp_verb = line
        else:
            tmp_prep = line.split(',')
            my_dict[tmp_verb]=tmp_prep

    json_list.append(my_dict)
    with open("parser.json","w") as pfile:
        json.dump(json_list,pfile, indent = 4)
#clean_json
import json

def read_json(filename):
    with open(filename, "r") as jfile:
        content = json.load(jfile)
        return content

jsonResult = read_json("final_out_parser.json")[0]
#print(type(jsonResult))
newDict = {}
newOutList = []
for key in jsonResult:
    value = jsonResult[key]
    instrcNum = len(value)
    delteList = []
    newList = []
    for i in range(0,instrcNum):
        for j in range(0,instrcNum):
            if value[j] in value[i] and value[j] != value[i]:
                delteList.append(j)
    for i in range(0,instrcNum):
        if i not in delteList:
            newList.append(value[i])
    #print(key)
    #print(value)
    #print(delteList)
    #exit()
    newDict[key] = newList
newOutList.append(newDict)
with open("new_out_parser.json","w") as pfile:
    json.dump(newOutList,pfile, indent = 4)




        

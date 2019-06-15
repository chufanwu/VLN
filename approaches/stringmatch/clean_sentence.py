import json

def clean_sentence(s):
    def remove_char(s, c):
        while c in s:
            s = s.replace(c, '')
        return s

    s = remove_char(s, '(')
    s = remove_char(s, ')')

    # remove 'A' to 'Z'
    for i in range(65, 91):
        s = remove_char(s, chr(i))

    s = s.split(' ')
    while '' in s:
        s.remove('')
    while ' ' in s:
        s.remove(' ')

    s = ' '.join(s)

    return s


def find_type(s, target):
    result = []

    def find_type_once(s, target):
        oriStr = s
        idx = s.index(target)
        s = s[idx + len(target):]

        bracketCnt = 1
        resultStr = ''
        while bracketCnt != 0:
            c = s[0]
            resultStr += c
            s = s[1:]

            if c == '(':
                bracketCnt += 1
            elif c == ')':
                bracketCnt -= 1

        resultStr = resultStr[:-1]
        nextStr = oriStr[:idx] + target.lower() + oriStr[idx+len(target):]

        return resultStr.strip(), nextStr

    while target in s:
        resultStr, s = find_type_once(s,target)
        result.append(resultStr)

    return result

def read_json(filename):
    with open(filename, "r") as jfile:
        content = json.load(jfile)
        return content

def nn_dict(inputString):
    addList = []
    npResult = find_type(inputString, 'NN ')
    for npItem in npResult:
        addList.append(npItem)
    npResult = find_type(inputString,'NNS ')
    for npItem in npResult:
        addList.append(npItem)
    return addList

def get_all_nouns():
    jsonResult = read_json('new_out_parser.json')[0]
    allDict = {}
    allList = []
    for key in jsonResult:
        value = jsonResult[key]
        for instrucString in value:
            tmpList = nn_dict(instrucString)
            for nnItem in tmpList:
                try:
                    allDict[nnItem] += 1
                except KeyError:
                    allDict[nnItem] = 1
    for key in allDict:
        if allDict[key]>2:
            allList.append(key)
    return allList

def list2file(addr, list):
    with open(addr, "w") as file:
        for item in list:
            file.write("%s\n"%item)

def write_dict_file():
    allList = get_all_nouns()
    list2file("dict.txt",allList)

def file2list(addr):
    with open(addr,"r") as file:
        content = file.readlines()
        content = [x.strip() for x in content]
        return content

def get_nouns_list():
    objList = []
    regionList = []
    locList = []
    allList = file2list("dict.txt")
    for line in allList:
        if line[0]=='!':
            regionList.append(line[1:])
        elif line[0]=='@':
            locList.append(line[1:])
        else:
            objList.append(line)
    return objList,regionList,locList


def join_nn_str(inputString):
    #join '(DT the) (NN right)' as (NN the_right)
    joinStr = ''
    for wordItem in inputString:
        if wordItem.islower() or wordItem == '(':
            joinStr += wordItem
    #  '(the(right'
    newStr = '_'.join(joinStr[1:].split('('))
    return newStr + ')'
    # return '(NN '+newStr + ')'

def wcf_clean_sentence():
    jsonResult = read_json('new_out_parser.json')[0]
    objList,regionList,locList = get_nouns_list()
    newDict = {}
    for key in jsonResult:
        value = jsonResult[key] # value is a list of instructions
        for instrucString in value: #every VP
            newInstrucStr = instrucString
            npList = find_type(instrucString,'NP ')
            for npItem in npList:
                if "NP" not in npItem:
                    newNpItem = join_nn_str(npItem)
                    replaceFlag = True
                    nnItem1 = find_type(npItem,'NN ')
                    nnItem2 = find_type(npItem,'NNS ')
                    nnItem = nnItem1 + nnItem2
                    if len(nnItem) == 0:
                        replaceFlag=False
                    elif nnItem[-1] in objList:
                        newNpItem = '(NN OBJ_' + newNpItem
                    elif nnItem[-1] in regionList:
                        newNpItem = '(NN REGION_' + newNpItem
                    elif nnItem[-1] in locList:
                        newNpItem = '(NN LOCATION_' + newNpItem
                    else:
                        replaceFlag = False
                    if replaceFlag == True:
                        newInstrucStr = newInstrucStr.replace(npItem,newNpItem)
                    #replace the npItem corrsponding to the nnItem
            try:
                newDict[key].append(newInstrucStr)
            except KeyError:
                newDict[key] = []
                newDict[key].append(newInstrucStr)
    newList = []
    newList.append(newDict)
    with open("new_new_out_parser.json","w") as pfile:
        json.dump(newList,pfile, indent = 4)




if __name__ == '__main__':
    #write_dict_file()
    #print(join_nn_str("(DT the) (NN right)"))
    wcf_clean_sentence()
            
#s = '(ROOT(S(VP (VB go) (PP (TO to)(NP(NP (DT the) (NN right))(PP (IN of)(NP (DT the) (NN sofa))))))))'
#result = find_type(s,'NN')
#print (result)
#['right', 'sofa']
#['(NP (DT the) (NN right))(PP (IN of)(NP (DT the) (NN sofa)))', '(DT the) (NN right)', '(DT the) (NN sofa)']
#(NP(NN()))
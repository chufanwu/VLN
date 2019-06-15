#getting the parser tree for each element
import os
from os.path import join
import time

parserPath = '/home/chufanw/parser/stanford/stanford-parser-full-2018-02-27/'

def str2file(addr, inputString):
    with open(addr, "w") as file:
        file.write("%s\n"%inputString)

def list2file(addr, list):
    with open(addr, "w") as file:
        for item in list:
            file.write("%s\n"%item)

def file2list(addr):
    with open(addr,"r") as file:
        content = file.readlines()
        content = [x.strip() for x in content]
        return content

#copyright stackexchange https://codereview.stackexchange.com/questions/146834/function-to-find-all-occurrences-of-substring
def substring_indexes(substring, string):
    """ 
    Generate indices of where substring begins in string

    >>> list(find_substring('me', "The cat says meow, meow"))
    [13, 19]
    """
    last_found = -1  # Begin at -1 so the next position to search from is 0
    while True:
        # Find next index of substring, by starting after its last known position
        last_found = string.find(substring, last_found + 1)
        if last_found == -1:  
            break  # All occurrences have been found
        yield last_found

def get_parser():
    allLongList = file2list("all_long.txt")
    FinalOutDict = {}
    FinalOutList = []
    indCount = 0
    allCount = len(allLongList)
    for longInstrc in allLongList:
        indCount += 1
        if indCount %200 == 1:
            startTime = time.time()
        str2file("tmp_parser.txt",longInstrc)
        cmd = join(parserPath,'lexparser.sh')+' '+"tmp_parser.txt" + ' >tmp_out.txt'
        os.system(cmd)
        outParser = file2list('tmp_out.txt')
        strOutParser = ' '.join(outParser)
        vbPositions = substring_indexes("VB",strOutParser)
        vbList = []
        for item in vbPositions:
            vbList.append(item)
        #print "printing the vblist"
        #print vbList
        #print strOutParser[vbList[0]:vbList[1]]
        outList = []
        for vbPos in vbList:
            if strOutParser[vbPos-5:vbPos-2] == '(VP':
                startPos = vbPos-5
                count = 0
                index = -1
                for i in strOutParser[startPos:]:
                    index += 1
                    if i == '(':
                        count += 1
                    elif i == ')':
                        count -= 1
                    if count == 0:
                        break
                tmpStr = strOutParser[startPos:startPos+index+1]
                outList.append(tmpStr)
        #FinalOutDict[longInstrc] = outList
        FinalOutDict[longInstrc] = strOutParser

        
        
        if indCount % 200 == 0:
            pastTime = time.time() - startTime
            print "Finished %.4f percent, using %d second" %(float(indCount)/allCount, pastTime)



    FinalOutList.append(FinalOutDict)
    with open("final_out_parser.json","w") as pfile:
        json.dump(FinalOutList,pfile, indent = 4)

if __name__ == '__main__':
    get_parser()
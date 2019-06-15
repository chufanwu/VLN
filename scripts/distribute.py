# calculate the distribution of the dataset
import os
import os.path as osp
import json
from textblob import TextBlob
from collections import Counter

def read_json(filename):
    with open(filename, "r") as jfile:
        content = json.load(jfile)
        return content



if __name__ == '__main__':
    
    file_path = 'tasks/R2R/data/R2R_test.json'
    content = read_json(file_path)
    # print (len(content))  #4675 train 
    ignore_list = ['right','left','walk','turn','wait','exit','stop']
    nouns = []
    for item in content:
        for command in item['instructions']:
               blob = TextBlob(command)
               for phrase in blob.tags:
                   if phrase[1] in ['NN','NNS'] and phrase[0] not in ignore_list:
                        nouns.append(phrase[0].lower())
    ans = Counter(nouns).most_common(10)

    



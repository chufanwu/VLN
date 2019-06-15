# for generating sentences
import json
from os import path
from collections import Counter

base_vocab = ['<PAD>', '<UNK>', '<EOS>']
padding_idx = base_vocab.index('<PAD>')
roomString = 'LivingRoom/LivingArea/Bedroom/Bathroom/DiningRoom/LaudryRoom/MeetingRoom/Kitchen/Office/Closet/Garage/Library/Balcony/ClassRoom/Bar/Porch/Outdoor/Hallway/Entryway/Exercise/ExerciseRoom'
verbString = 'Turn left/Turn right/Go forward/Go backward/Walk'
relationString = 'through/at/pass/across/towards'

def save2json(inList, outName):
    '''
        save a list to json 
        input: a list, a saving name
        output: None
    '''
    with open(path.join("data", outName +'.json'), "w") as jfile:
        json.dump(inList, jfile, indent = 4)

def read_json(filename):
    with open(filename, "r") as jfile:
        content = json.load(jfile)
        return content

def label_prases():
    roomList = [x.lower() for x in roomString.split('/')]
    save2json(roomList, "roomList")
    
    verbList = [x.lower() for x in verbString.split('/')]
    save2json(verbList, "verbList")
    
    relationList = [x.lower() for x in relationString.split('/')]
    save2json(relationList, "relationList")


def gen_sentence():
    verbList = read_json("data/verbList.json")
    relationList = read_json("data/relationList.json")
    roomList = read_json("data/roomList.json")
    sentences = []

    for verb in verbList:
        for relation in relationList:
            for room in roomList:
                sentence = verb.split(' ') + relation.split(' ') + ['the'] + room.split(' ')
                sentence = " ".join(sentence)
                sentences.append(sentence)
    save2json(sentences, "sentences")   

def build_vocab(min_count=5, start_vocab=base_vocab):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    data = read_json('data/sentences.json')
    for item in data:
        count.update(item.split(' '))
    vocab = list(start_vocab)
    for word,num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    save2json(vocab,"vocab")

if __name__ == '__main__':
    # label_prases()
    # gen_sentence()
    build_vocab()





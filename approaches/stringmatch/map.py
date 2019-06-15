# mapping.py
import json
import re
import string
from collections import Counter
# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>']
padding_idx = base_vocab.index('<PAD>')


class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character
  
    def __init__(self, vocab=None, encoding_length=20):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i

    def split_sentence(self, sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def encode_sentence(self, sentence):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []
        for word in self.split_sentence(sentence)[::-1]: # reverse input sentences
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(self.word_to_index['<UNK>'])
        encoding.append(self.word_to_index['<EOS>'])
        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length-len(encoding))
        return np.array(encoding[:self.encoding_length])

    def decode_sentence(self, encoding):
        sentence = []
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.vocab[ix])
        return " ".join(sentence[::-1]) # unreverse before output


def file2str(addr):
    with open(addr,"r") as file:
        content = file.readlines()
        content = [x.strip() for x in content]
        return content

def str2file(addr, list):
    with open(addr, "w") as file:
        for item in list:
            file.write("%s\n"%item)

def sub_embedding():
    scan_c = file2str("category_vocab.txt")
    scan_e = file2str("embedding_vocab.txt")
    # print scan_c[:10]
    # print scan_e[:10]
    match_list = [x for x in scan_e if x in scan_c]
    rest_list = [x for x in scan_e if x not in scan_c]
    print("scan_c",len(scan_c),"scan_e",len(scan_e),"match_ls", len(match_list),"rest_list",len(rest_list))
    # str2file("embedding", rest_list)
    print(match_list)

def read_json(filename):
    with open(filename, "r") as jfile:
        content = json.load(jfile)
        return content

def read_join_words(filename):
    names = file2str(filename)
    join_names = []
    for name in names:
        if '-' in name:
            join_names.append(" ".join(name.split('-')))
    return join_names

def get_join_traingset():   # get the new training set by replacing the join item
    join_names = read_join_words("category_vocab.txt")
    # a = ['dwed','de','rgrg']
    # print (" ".join(a))
    # print (join_names)
    data_train = file2str("R2R_train.json")
    with open("R2R_train_new.json","w") as rfile:
        for line in data_train:
            line = line.lower()
            for name in join_names:
                if name.lower() in line:
                    line = line.replace(name.lower(), ''.join(name.lower().split(' ')))
            rfile.write("%s\n"%(line))

def load_datasets(splits):
    data = []
    for split in splits:
        assert split in ['train', 'val_seen', 'val_unseen', 'test']
        with open('R2R_%s_new.json' % split) as f:
            data += json.load(f)
    return data

def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    t = Tokenizer()
    data = load_datasets(splits)
    for item in data:
        for instr in item['instructions']:
            count.update(t.split_sentence(instr))
    vocab = list(start_vocab)
    for word,num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab

def deletejoin():
    categ = file2str("category_vocab.txt")
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character
    with open("category_vocab_new.txt","w") as cfile:
        for item in categ:
            tmp_item = SENTENCE_SPLIT_REGEX.sub('',item)
            cfile.write("%s\n"%(tmp_item))


if __name__ == '__main__':
    vocab = build2_vocab()
    str2file("embedding_vocab_new.txt",vocab)
    # get_join_traingset()
    # deletejoin()
    # get_join_traingset()


    

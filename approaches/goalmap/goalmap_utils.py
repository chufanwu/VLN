import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import re
import string
import sys


def LocationSign(x, z, radius=1.0, dense=50):
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

    # color = np.ones((coor.shape[0], 3)).astype('uint8')
    # color[:, 0] = 255

    return coor


def generateTopView(features, depth, size=28, heightThreshold=0.7):
    # features bs * 1025 * 28 * 28, cuda tensor
    # depth bs * 1 * 28 * 28, numpy array

    batchSize, channel, H, W = features.size()
    CAMERA_FACTOR = 4000.0
    topViewSize = size
    imageScale = H / 640.0
    cx = 320.0 * imageScale
    cy = 320.0 * imageScale
    fx = 320.0 * imageScale
    fy = 320.0 * imageScale

    # Cut off depth()
    mask = (depth >= 2.5 * depth.mean())
    depth[mask] = 0.0
    # depth = np.squeeze(depth, 1)
    # Calculate the coordinates
    pz = depth * 65535 / CAMERA_FACTOR

    mesh_px = np.repeat(np.array(range(0, W)).reshape(1, -1), H, axis=0).reshape(1, H, W)
    mesh_px = np.repeat(mesh_px, batchSize, axis=0).astype('float32')
    px = (mesh_px - cx) * pz / fx

    mesh_py = range(0, H)
    mesh_py.reverse()
    mesh_py = np.repeat(np.array(mesh_py).reshape(-1, 1), W, axis=1).reshape(1, H, W)
    mesh_py = np.repeat(mesh_py, batchSize, axis=0).astype('float32')
    py = (mesh_py - cy) * pz / fy

    # pz = -1 * pz

    # py = -1 * py

    px = px.reshape(batchSize, H, W, 1)
    py = py.reshape(batchSize, H, W, 1)
    pz = pz.reshape(batchSize, H, W, 1)

    coor = np.concatenate((px, py, pz), axis=3)
    coor = coor.reshape(batchSize, -1, 3)  # batchSize * 784 * 3

    features = features.contiguous().view(batchSize, channel, -1)  # batch * 1025 *  784
    features = torch.transpose(features, 2, 1)  # batch * 784 * 1025

    # add ego-location sign
    locCoor = LocationSign(0.0, 0.0, radius=0.5, dense=20)
    locCoor = locCoor.reshape(1, -1, 3)
    locCoor_len = locCoor.shape[1]
    locCoor = np.repeat(locCoor, batchSize, axis=0)  # batch * locCoor_len * 3
    coor = np.concatenate((coor, locCoor), axis=1)


    x_min = np.min(coor[:, :, 0], axis=1).reshape(batchSize, 1)
    y_min = np.min(coor[:, :, 1], axis=1).reshape(batchSize, 1)
    z_min = np.min(coor[:, :, 2], axis=1).reshape(batchSize, 1)

    coor[:, :, 0] -= x_min
    coor[:, :, 1] -= y_min
    coor[:, :, 2] -= z_min

    x_max = np.max(coor[:, :, 0], axis=1).reshape(batchSize, 1)
    # y_max = np.max(coor[:, :, 1], axis=1).reshape(batchSize, 1)
    z_max = np.max(coor[:, :, 2], axis=1).reshape(batchSize, 1)

    for i in range(batchSize):
        if x_max[i] > z_max[i]:
            zoom_in = (topViewSize - 2) / x_max[i]
        else:
            zoom_in = (topViewSize - 2) / z_max[i]
        coor[i, :, :] *= zoom_in

    # y_size = np.max(coor[:, :, 1], axis=1).reshape(batchSize, 1)

    coor = np.floor(coor).astype('int32')

    topView = np.zeros((1, topViewSize, topViewSize, channel + 1)).astype(
        'float32')  # channel + 1, since we add a channel indicate ego-location
    topView = np.repeat(topView, batchSize, axis=0)
    topView = Variable(torch.from_numpy(topView))  # batch * 28 * 28 * 1026
    if torch.cuda.is_available():
        topView = topView.cuda()

    # HEIGHT_TRESHOLD = y_size * heightThreshold

    egoLoc_coor = coor[:, -locCoor_len:, :]
    coor = coor[:, :-locCoor_len, :]

    for i in range(batchSize):
        x = coor[i, :, 0].reshape(-1)
        # y = coor[i,:, 1].reshape(-1)
        z = coor[i, :, 2].reshape(-1)

        ego_x = egoLoc_coor[i, :, 0].reshape(-1)
        ego_z = egoLoc_coor[i, :, 2].reshape(-1)

        ii = torch.LongTensor([i])
        x = torch.from_numpy(x).long()
        z = torch.from_numpy(z).long()
        z = topViewSize - 2 - z

        ego_x = torch.from_numpy(ego_x).long()
        ego_z = torch.from_numpy(ego_z).long()
        ego_z = topViewSize - 2 - ego_z

        topView[ii:ii + 1, z, x, :-1] = features[ii:ii + 1, :, :]
        topView[ii:ii + 1, ego_z, ego_x, -1] = 1.0

    topView = topView.transpose(1, 3)
    topView = topView.transpose(2, 3)

    # for i in range(0,128):
    #     plt.imshow(topView[0,i,:,:].data.numpy() * 255)
    #     plt.ioff()
    #     plt.show()

    return topView  # batch * (1024+2) * 28 * 28


def buildObjectCategory(path):
    category = {}
    with open(path, 'r') as fp:
        for line in fp:
            if line[0] == 'i':
                continue

            item = line.split('\t')
            if item[7] == '':
                item[7] = item[2]
            category[int(item[0])] = item[7].replace(' ', '-')
            # category.append([item[0],item[1],item[2],item[7]])

    category[0] = 'unknown'

    return category


def buildRegionCategory(path):
    category = {}
    with open(path, 'r') as fp:
        for line in fp:
            num = ord(line[1])
            name = line[6:-1] if line[-1] == '\n' else line[6:]
            category[num] = name.replace(' ', '-')
    return category


def buildVocaulary(regionCategoryPath, objectCategoryPath):
    regionCategory = buildRegionCategory(regionCategoryPath)
    objectCategory = buildObjectCategory(objectCategoryPath)

    words = [value for (key, value) in regionCategory.items()] + [value for (key, value) in objectCategory.items()]
    words = list(set(words))

    vocabulary = {}
    vocabularyInverse = {}
    for idx, word in enumerate(words):
        vocabulary[word] = idx
        vocabularyInverse[idx] = word

    return vocabulary, vocabularyInverse


def buildEmbeddingVocabulary(path):
    voc = []
    with open(path, 'r') as fp:
        for line in fp:
            if '\n' in line:
                line = line[:-1]
            voc.append(line)

    return voc


def sort_batch(instr_encoding):
    ''' Extract instructions from a list of observations and sort by descending
        sequence length (to enable PyTorch packing). '''
    base_vocab = ['<PAD>', '<UNK>', '<EOS>']
    padding_idx = base_vocab.index('<PAD>')

    seq_tensor = instr_encoding
    seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
    seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length

    seq_lengths = torch.from_numpy(seq_lengths).long()

    # Sort sequences by lengths
    seq_lengths, perm_idx = seq_lengths.sort(0, True)

    # return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
    #        mask.byte().cuda(), \
    #        list(seq_lengths), list(perm_idx)
    return seq_lengths, perm_idx


class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')  # Split on any non-alphanumeric character

    def __init__(self, vocab=None, encoding_length=20):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        if vocab:
            for i, word in enumerate(vocab):
                self.word_to_index[word] = i

    def split_sentence(self, sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if
                     len(s.strip()) > 0]:
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
        for word in self.split_sentence(sentence[::1]):  # not reverse input sentences
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(self.word_to_index['<UNK>'])
        encoding.append(self.word_to_index['<EOS>'])
        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length - len(encoding))
        return np.array(encoding[:self.encoding_length]).reshape(1,-1)

    def decode_sentence(self, encoding):
        sentence = []
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.vocab[ix])
        return " ".join(sentence[::1])  # not unreverse before output

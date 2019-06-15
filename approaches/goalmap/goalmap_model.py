import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from goalmap_utils import generateTopView

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'http://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

base_vocab = ['<PAD>', '<UNK>', '<EOS>']
padding_idx = base_vocab.index('<PAD>')


class MyResNet101_FasterRCNN(ResNet):
    def __init__(self, modelPath='/home/qiyuand/FasterRCNN/v1/pretrained_model/res101_faster_rcnn_iter_1190000.pth',
                 dropout=False):
        super(MyResNet101_FasterRCNN, self).__init__(Bottleneck, [3, 4, 23, 3], 1000)
        self.dropout = dropout
        self.dropout2d = nn.Dropout2d(p=0.5)
        del self.fc

        cp = torch.load(modelPath)
        d = {k: cp['resnet.' + k] for k in list(self.state_dict().keys()) if 'fc' not in k}
        self.load_state_dict(d)

    def forward(self, x):  # bs * 3 * 224 * 224

        x = self.conv1(x)  # bs * 64 * 112 * 112
        if self.dropout:
            x = self.dropout2d(x)
        x = self.bn1(x)  # bs * 64 * 112 * 112
        x = self.relu(x)  # bs * 64 * 112 * 112
        x = self.maxpool(x)  # bs * 64 * 56 * 56

        x = self.layer1(x)  # bs * 256 * 56 * 56
        if self.dropout:
            x = self.dropout2d(x)
        x = self.layer2(x)  # bs * 512 * 28 * 28
        x = self.layer3(x)  # bs * 1024 * 14 * 14
        # x = self.layer4(x)  # bs * 512 * 20 * 20

        # x = self.avgpool(x) # bs * 512 * 14 * 14
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


class Map(nn.Module):
    # the rest of the network
    # SM: Semantic Map
    # WE: Word Embedding
    # f_depth is the feature depth i.e. channel numbers for the two maps
    # remember to wrap all the torch tensor to Variables
    # won't change the dim of sm
    def __init__(self, wordEmbeddingNum, wordEmbeddingDim, hidden_size=256, padding_idx=padding_idx, dropout_ratio=0.5,
                 featureChannel=1024,
                 releMapChannel=256, goalMapChannel=64, dropout=True):
        super(Map, self).__init__()
        # w_out_dim is the feature size for each kernel (depth into the channel)
        self.wordEmbeddingDim = wordEmbeddingDim
        self.embedding = EncoderLSTM(wordEmbeddingNum, wordEmbeddingDim, hidden_size, padding_idx,
                                     dropout_ratio, bidirectional=False, num_layers=1)
        self.featureChannel = featureChannel
        self.releMapChannel = releMapChannel
        self.goalMapChannel = goalMapChannel
        self.linearLayer1 = torch.nn.Linear(hidden_size, releMapChannel * featureChannel * 1 * 1, bias=True)
        self.linearLayer3 = torch.nn.Linear(hidden_size, goalMapChannel * (featureChannel + 2) * 3 * 3, bias=True)

        self.releMapBN = nn.BatchNorm2d(releMapChannel)
        self.goalMapBN = nn.BatchNorm2d(goalMapChannel)

        self.leakyRelu = torch.nn.LeakyReLU(negative_slope=0.01)
        self.dropoutLayer = nn.Dropout(p=0.3)
        self.dropout = dropout

    def forward(self, feature, depth, instruction_idx,instruction_length, only_train_rele):
        batchSize = feature.size(0)
        assert feature.size(0) == instruction_idx.size(0)
        H = feature.size(2)
        W = feature.size(3)
        assert H == W

        intructionEmbedding = self.embedding(instruction_idx,instruction_length)  # batch * hidden_size

        # compute the conv kernel of relevance map
        filter1 = self.leakyRelu(self.linearLayer1(intructionEmbedding))
        if self.dropout:
            filter1 = self.dropoutLayer(filter1)
        filter1 = filter1.view(batchSize, self.releMapChannel, self.featureChannel, 1, 1)

        # relevance map conv
        relevanceMap = None  # keep the spatial size
        for i in range(0, batchSize):
            _feature = feature[i].contiguous().view(1, self.featureChannel, H, W)
            _relevanceMap = F.relu(F.conv2d(_feature, filter1[i]))
            # _relevanceMap = _relevanceMap / torch.max(_relevanceMap)

            if relevanceMap is None:
                relevanceMap = _relevanceMap
            else:
                relevanceMap = torch.cat((relevanceMap, _relevanceMap), dim=0)
        relevanceMap = self.releMapBN(relevanceMap)
        relevanceMap = relevanceMap.view(batchSize, self.releMapChannel, H * W)
        singleRelevanceMap, idx = torch.max(relevanceMap, dim=1)
        singleRelevanceMap = singleRelevanceMap.view(batchSize, H, W)
        singleRelevanceMap = torch.clamp(singleRelevanceMap,min=0.0,max=1.0)

        del instruction_idx,relevanceMap

        if only_train_rele:
            return singleRelevanceMap

        # use depth image to compute the top view of feature and relevance map for the following goal map
        feature_releMap = torch.cat((feature, singleRelevanceMap.view(batchSize,1,H,W)), dim=1)  # batch * 1025 * 28 * 28
        topView_feature_releMap = generateTopView(feature_releMap, depth,
                                                  size=H)  # batch * 1025 * 28 * 28, depth is numpy with size bs * 1 * 28 * 28
        singleRelevanceMap_topView = topView_feature_releMap[:, -2, :, :]  # batch * 28 * 28, for return

        # topView_feature_releMap[:, -1, :, :] =  singleRelevanceMap_topView # remove the ego location

        # compute the conv kernel
        filter3 = self.leakyRelu(self.linearLayer3(intructionEmbedding))
        if self.dropout:
            filter3 = self.dropoutLayer(filter3)
        filter3 = filter3.view(batchSize, self.goalMapChannel, self.featureChannel + 2, 3, 3)

        # goal map conv
        goalMap = None  # keep the spatial size
        for i in range(0, batchSize):
            _feature = topView_feature_releMap[i].contiguous().view(1, self.featureChannel + 2, H, W)
            _goalMap = F.relu(F.conv2d(_feature, filter3[i],padding=1,stride=1))
            # _goalMap = _goalMap / torch.max(_goalMap)

            if goalMap is None:
                goalMap = _goalMap
            else:
                goalMap = torch.cat((goalMap, _goalMap), dim=0)
        goalMap = self.goalMapBN(goalMap)
        goalMap = goalMap.view(batchSize, self.goalMapChannel, H * W)
        singleGoalMap, idx = torch.max(goalMap, dim=1)
        singleGoalMap = singleGoalMap.view(batchSize, H, W)
        singleGoalMap = torch.clamp(singleGoalMap,min=0.0,max=1.0)


        return singleRelevanceMap, singleRelevanceMap_topView, singleGoalMap  # batch * 28 * 28


class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                 dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
                                         hidden_size * self.num_directions
                                         )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        if torch.cuda.is_available():
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a 
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]

        embedding = nn.Tanh()(self.encoder2decoder(h_t))

        return embedding
        # (batch, hidden_size)

        # if __name__ == '__main__':
        # feature = Variable(torch.rand(4, 128, 80, 80))
        # wordEm = Variable(torch.rand(4, 256))
        #
        # Rele = MyResNet101_FasterRCNN(wordEmbeddingDim=256, featureChannel=128, mapChannel=128)
        # r = Rele(feature, wordEm)
        #
        # print r.size()

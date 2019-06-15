import os

import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from goalmap_model import Map, MyResNet101_FasterRCNN
from goalmap_utils import buildVocaulary, buildEmbeddingVocabulary, Tokenizer, sort_batch
from goalmap_dataset_relemap_sentence import DatasetGenerator


class Task():
    def __init__(self, args):
        print '#' * 60
        print ' ' * 20 + '    Task Created    ' + ' ' * 20
        print '#' * 60

        ######################################################################################################
        # Basic parameters
        self.batchSize = args.batchSize
        self.lrCNN = args.lrCNN
        self.lrRele = args.lrRele

        self.weightDecay = 1e-4
        self.wordEmbeddingDim = 256
        self.featureMapChannel = 1024
        self.instructionLength = 10
        self.pinMemory = True
        self.dropout = False

        self.lossAll = []
        self.lossAllEpoch = []
        self.lossInEpoch = []
        self.lossRecent = []
        self.lossRecentRange = 100
        self.lossRecentPlot = []
        self.testLossInEpoch = []
        self.testLossAllEpoch = []
        self.epoch = args.epoch
        self.epoch_i = 0

        self.batchPerlossDisplay = args.batchPerlossDisplay
        self.batchPerVisualize = args.batchPerVisualize
        self.batchPerPlot = args.batchPerPlot
        self.batchPerModelSave = args.batchPerModelSave
        self.saveFigure = True
        self.visualize = True
        self.checkPoint = args.checkPoint

        # Path
        self.objectCategoryPath = './data/category_mapping.tsv'
        self.regionCategoryPath = './data/region_category.txt'
        self.trainSplitPath = './dataset_split/train_scans.txt'
        self.testSplitPath = './dataset_split/test_scans.txt'
        self.embeddingVocPath = './data/embedding_vocab.txt'

        self.imageRootPath = args.imageRootPath
        self.logPath = args.logPath
        self.cnnpretrainedPath = '/home/qiyuand/FasterRCNN/v1/pretrained_model/res101_faster_rcnn_iter_1190000.pth'

        ######################################################################################################

        # Dataset
        self.vocabulary, self.vocabularyInv = buildVocaulary(self.regionCategoryPath, self.objectCategoryPath)
        self.embeddingVoc = buildEmbeddingVocabulary(self.embeddingVocPath)

        self.tokenizer = Tokenizer(vocab=self.embeddingVoc, encoding_length=self.instructionLength)

        self.trainDataset = DatasetGenerator(scanListPath=self.trainSplitPath, ImageRootPath=self.imageRootPath,
                                             Vocaubulary=self.vocabulary, depth=False)
        self.testDataset = DatasetGenerator(scanListPath=self.testSplitPath, ImageRootPath=self.imageRootPath,
                                            Vocaubulary=self.vocabulary, depth=False)
        self.trainDataLoader = DataLoader(dataset=self.trainDataset, shuffle=True, batch_size=self.batchSize,
                                          num_workers=24, pin_memory=self.pinMemory)
        self.testDataLoader = DataLoader(dataset=self.testDataset, shuffle=True, batch_size=self.batchSize,
                                         num_workers=24, pin_memory=self.pinMemory)
        print 'Dateset Loaded.'

        # Create model
        self.rgbResNetModel = MyResNet101_FasterRCNN(dropout=self.dropout, modelPath=self.cnnpretrainedPath)
        self.map = Map(wordEmbeddingNum=len(self.embeddingVoc),
                       wordEmbeddingDim=self.wordEmbeddingDim,
                       dropout=self.dropout,
                       featureChannel=self.featureMapChannel)

        # Run task on all available GPUs
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print("Use ", torch.cuda.device_count(), " GPUs!")

                self.rgbResNetModel = nn.DataParallel(self.rgbResNetModel)
                self.map = nn.DataParallel(self.map)
            self.rgbResNetModel = self.rgbResNetModel.cuda()
            self.map = self.map.cuda()
            print 'Model Created on GPUs.'

        # Optimizer
        self.rgbOptimizer = optim.Adam(self.rgbResNetModel.parameters(), lr=self.lrCNN, betas=(0.9, 0.999), eps=1e-08,
                                       weight_decay=self.weightDecay)
        self.mapOptimizer = optim.Adam(self.map.parameters(), lr=self.lrRele, betas=(0.9, 0.999),
                                       eps=1e-08,
                                       weight_decay=self.weightDecay)
        # Scheduler
        self.rgbScheduler = ReduceLROnPlateau(self.rgbOptimizer, factor=0.1, patience=10, mode='min')
        self.mapScheduler = ReduceLROnPlateau(self.mapOptimizer, factor=0.1, patience=10, mode='min')

        # Loss Function
        self.loss = torch.nn.MSELoss()

        # Load model given a checkPoint
        if self.checkPoint != "":
            self.load(self.checkPoint)

    def train(self):

        print 'Training task begin.'
        print '----Batch Size: %d' % self.batchSize
        print '----Learning Rate: %f, %f' % (self.lrCNN, self.lrRele)
        print '----Epoch: %d' % self.epoch
        print '----Log Path: %s' % self.logPath

        print

        for self.epoch_i in range(self.epoch):

            if self.epoch_i == 0:
                self.save(batchIdx=0)  # Test the save function

            # if self.epoch_i != 0:
            #     try:
            #         self.rgbResNetModel = self.rgbResNetModel.eval()
            #         self.map = self.map.eval()
            #         self.test()
            #     except Exception, e:
            #         print e

            self.rgbResNetModel = self.rgbResNetModel.train()
            self.map = self.map.train()
            self.epochTrain()
            self.save()

    def epochTrain(self):
        self.lossInEpoch = []
        s = '#' * 30 + '    Epoch %3d / %3d    ' % (self.epoch_i + 1, self.epoch) + '#' * 30
        print s

        for idx, (rgb, releTarget, instruction) in enumerate(self.trainDataLoader):
            # rgb: batch * 3 * 224 * 224, tensor
            # depth: batch * 1 * 28 * 28, numpy
            # releTarget: batch * 1 * 28 * 28, tensor
            # goalTarget: batch * 1 * 28 * 28, tensor
            # instruction: batch * lens, string


            if torch.cuda.is_available():
                torch.cuda.empty_cache()


            # First sort the batch according to the length of instruction
            batchSize = rgb.size(0)
            instruction_idx = None
            for i in range(batchSize):
                if instruction_idx is None:
                    instruction_idx = self.tokenizer.encode_sentence(instruction[i])
                else:
                    instruction_idx = np.concatenate((instruction_idx, self.tokenizer.encode_sentence(instruction[i])),
                                                     axis=0)

            seq_lengths, perm_idx = sort_batch(instruction_idx)  # input in numpy and return in tensor

            # conver to Variable
            rgb = Variable(rgb)
            releTarget = Variable(releTarget, requires_grad=False)
            # goalTarget = Variable(goalTarget, requires_grad=False)
            releTarget = releTarget.squeeze(1)
            # goalTarget = goalTarget.squeeze(1)
            instruction_idx = Variable(torch.from_numpy(instruction_idx).long())
            seq_lengths = Variable(seq_lengths.long())

            if torch.cuda.is_available():
                rgb = rgb.cuda()
                releTarget = releTarget.cuda()
                # goalTarget = goalTarget.cuda()
                instruction_idx = instruction_idx.cuda()
                perm_idx = perm_idx.cuda()
                seq_lengths = seq_lengths.cuda()

            # sort according the length
            rgb = rgb[perm_idx]
            releTarget = releTarget[perm_idx]
            # goalTarget = goalTarget[perm_idx]
            instruction_idx = instruction_idx[perm_idx]


            # Go through the models
            feature = self.rgbResNetModel(rgb)  # 1024 * 14 * 14
            releMap = self.map(feature, None, instruction_idx,seq_lengths , True)  # depth in numpy
            lossValue = self.loss(input=releMap, target=releTarget)

            # Record the loss
            self.lossInEpoch.append(lossValue.data)
            self.lossAll.append(lossValue.data)
            if len(self.lossRecent) == self.lossRecentRange:
                self.lossRecent.pop(0)
            self.lossRecent.append(lossValue.data)
            self.lossRecentPlot.append(np.mean(self.lossRecent))

            # # Visualization
            try:
                if (idx + 1) % self.batchPerVisualize == 0 and self.visualize:
                    rgb_v = rgb[0, :, :, :].detach().cpu().numpy()
                    target_v = releTarget[0, :, :].detach().cpu().numpy()
                    category_v = instruction_idx[0].detach().cpu().numpy()
                    category_v = self.tokenizer.decode_sentence(category_v)
                    self.visualization(rgb_v, target_v, releMap[0, :, :].detach().cpu().numpy(), category_v, idx + 1)
            except Exception, e:
                print e


            # Display loss
            if (idx + 1) % self.batchPerlossDisplay == 0:
                print 'Epoch: %3d / %3d        Batch: %5d / %5d        BatchLoss: %6.4f        EpochAvgLoss: %6.4f' \
                      % (self.epoch_i + 1, self.epoch, idx + 1, np.ceil(len(self.trainDataset) / float(self.batchSize)),
                         self.lossInEpoch[-1], np.mean(self.lossInEpoch))

            del feature, releMap, rgb, releTarget, instruction_idx, seq_lengths, perm_idx
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Backward
            self.rgbOptimizer.zero_grad()
            self.mapOptimizer.zero_grad()
            lossValue.backward()
            self.rgbOptimizer.step()
            self.mapOptimizer.step()

            del lossValue

            # Plot the loss curves
            if (idx + 1) % self.batchPerPlot == 0:
                self.plot()


            # Save model
            if (idx + 1) % self.batchPerModelSave == 0:
                self.save(batchIdx=(idx + 1))

            # Release memory

        # Record the average loss of the entire epoch
        self.lossAllEpoch.append(np.mean(self.lossInEpoch))

    def test(self):
        self.testLossInEpoch = []
        s = '#' * 15 + '  Epoch %3d / %3d Test ' % (self.epoch_i + 1, self.epoch) + '#' * 15
        print s

        for idx, (rgb, target, category) in enumerate(self.testDataLoader):

            torch.cuda.empty_cache()

            # For visualization
            try:
                if (idx + 1) % self.batchPerVisualize == 0 and self.visualize:
                    rgb_v = rgb[0, :, :, :]
                    target_v = target[0, 0, :, :]
                    category_v = self.vocabularyInv[category.data.numpy()[0]]
            except Exception, e:
                print e

            rgb = Variable(rgb).cuda()
            target = Variable(target, requires_grad=False).cuda()
            target = target.squeeze(1)
            category = Variable(category).cuda()

            feature = self.rgbResNetModel(rgb)
            releMap = self.map(feature, category)
            lossValue = self.loss(input=releMap, target=target)

            self.testLossInEpoch.append(lossValue.item())

            if (idx + 1) % self.batchPerlossDisplay == 0:
                print 'Test    Epoch: %3d / %3d        Batch: %5d / %5d        BatchLoss: %6.4f' \
                      % (self.epoch_i + 1, self.epoch, idx + 1, np.ceil(len(self.testDataset) / float(self.batchSize)),
                         self.testLossInEpoch[-1])

            # Visualization
            try:
                if (idx + 1) % self.batchPerVisualize == 0 and self.visualize:
                    self.visualization(rgb_v, target_v, releMap[0, :, :], category_v, idx + 1, test=True)
            except Exception, e:
                print e

            del lossValue, feature, releMap, rgb, target, category

        self.testLossAllEpoch.append(np.mean(self.testLossInEpoch))
        print 'Epoch: %3d / %3d        Test Loss: %6.2f' % (self.epoch_i + 1, self.epoch, self.testLossAllEpoch[-1])

    def save(self, batchIdx=None):
        dirPath = os.path.join(self.logPath, 'models')

        if not os.path.exists(dirPath):
            os.mkdir(dirPath)

        if batchIdx is None:
            path = os.path.join(dirPath, 'Epoch-%03d-end.pth.tar' % (self.epoch_i + 1))
        else:
            path = os.path.join(dirPath, 'Epoch-%03d-Batch-%04d.pth.tar' % (self.epoch_i + 1, batchIdx))

        torch.save({'epochs': self.epoch_i + 1,
                    'batch_size': self.batchSize,
                    'learning_rate_cnn': self.lrCNN,
                    'learning_rate_rele': self.lrRele,
                    'weight_dacay': self.weightDecay,
                    'rgbResnet_model_state_dict': self.rgbResNetModel.state_dict(),
                    'map_model_state_dict': self.map.state_dict(),
                    'training_loss': self.lossAll,
                    'training_loss_epoch': self.lossAllEpoch,
                    'test_loss': self.testLossAllEpoch,
                    'rgbResnet_optimizer': self.rgbOptimizer.state_dict(),
                    'map_optimizer': self.mapOptimizer.state_dict()},
                   path)
        print 'Training log saved to %s' % path

    def load(self, path):
        print 'Load model from: ' + path

        modelCheckpoint = torch.load(path)
        self.rgbResNetModel.load_state_dict(modelCheckpoint['rgbResnet_model_state_dict'])
        # self.map.load_state_dict(modelCheckpoint['map_model_state_dict'])
        # self.rgbOptimizer.load_state_dict(modelCheckpoint['rgbResnet_optimizer'])
        # self.relevanceOptimizer.load_state_dict(modelCheckpoint['releMap_optimizer'])

        # self.lossAll = modelCheckpoint['training_loss']
        # self.lossAllEpoch = modelCheckpoint['training_loss_epoch']
        # self.testLossAllEpoch = modelCheckpoint['test_loss']

        return modelCheckpoint

    def plot(self):
        plt.switch_backend('agg')
        plt.figure(figsize=(24, 12))

        plt.subplot(2, 2, 1)
        plt.plot(self.lossRecentPlot)
        plt.title('Training Average Loss of Recent %d Batch' % self.lossRecentRange)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        # plt.ylim([0, 1.2])
        plt.xlim([0, len(self.lossRecentPlot)])
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(self.lossAll)
        plt.title('Training Loss of All Batch')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        # plt.ylim([0, 1.2])
        plt.xlim([0, len(self.lossAll)])
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(self.lossAllEpoch)
        plt.title('Training Loss of All Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.ylim([0, 1.2])
        plt.xlim([1, max(len(self.lossAllEpoch), 1)])
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.plot(self.testLossAllEpoch)
        plt.title('Test Loss of All Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.ylim([0, 1.2])
        plt.xlim([0, max(len(self.testLossAllEpoch), 1)])
        plt.grid()

        if self.saveFigure:
            dirPath = os.path.join(self.logPath, 'curve')
            if not os.path.exists(dirPath):
                os.mkdir(dirPath)

            figPath = os.path.join(dirPath, 'Epoch%3d.png' % (self.epoch_i + 1))
            plt.savefig(figPath)

    def visualization(self, rgb_v, target_v, releMap, category_v, batchIdx, test=False):
        plt.switch_backend('agg')
        plt.figure(figsize=(35, 35), dpi=150)

        plt.figure()
        plt.suptitle(category_v, fontsize=16)

        plt.subplot(1, 3, 1)
        rgb_v = rgb_v
        rgb_v = np.transpose(rgb_v, (1, 2, 0))
        plt.imshow(rgb_v)

        plt.subplot(1, 3, 2)
        plt.imshow(target_v)

        plt.subplot(1, 3, 3)
        plt.imshow(releMap)

        dirPath = os.path.join(self.logPath, 'visualization')
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)

        if not test:
            figPath = os.path.join(dirPath, 'Epoch%03d-Batch%05d.png' % (self.epoch_i + 1, batchIdx))
        else:
            figPath = os.path.join(dirPath, 'Epoch%03d-Test-Batch%05d.png' % (self.epoch_i + 1, batchIdx))
        plt.savefig(figPath)

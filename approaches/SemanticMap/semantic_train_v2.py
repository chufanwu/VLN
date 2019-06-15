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

from semantic_model import MyResNet, DepthResNet, RelevanceMap, MyResNet_40
from semantic_utils import buildVocaulary, generateTopView
from semantic_dataset import DatasetGenerator


class SemanticMapTask():
    def __init__(self, args):
        print '#' * 60
        print ' ' * 20 + '    Task Created    ' + ' ' * 20
        print '#' * 60

        ######################################################################################################
        self.batchSize = args.batchSize
        self.lr = args.lr
        self.weightDecay = 1e-3
        self.wordEmbeddingDim = 256
        self.pinMemory = True

        self.lossAll = []
        self.lossAllEpoch = []
        self.lossInEpoch = []
        self.testLossInEpoch = []
        self.testLossAllEpoch = []
        self.testSuccessAllEpoch = []
        self.epoch = args.epoch
        self.epoch_i = 0

        # self.gpu = 1
        # os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % self.gpu
        self.batchPerlossDisplay = args.batchPerlossDisplay
        self.batchPerVisualize = args.batchPerVisualize
        self.batchPerPlot = args.batchPerPlot
        self.saveEpoch = args.saveEpoch
        self.saveFigure = True
        self.visualize = True
        self.checkPoint = args.checkPoint

        self.objectCategoryPath = './data/category_mapping.tsv'
        self.regionCategoryPath = './data/region_category.txt'
        self.trainSplitPath = './dataset_split/train_scans.txt'
        self.testSplitPath = './dataset_split/test_scans.txt'

        self.imageRootPath = args.imageRootPath
        self.logPath = args.logPath

        ######################################################################################################



        # Dataset
        self.vocabulary, self.vocabularyInv = buildVocaulary(self.regionCategoryPath, self.objectCategoryPath)
        self.trainDataset = DatasetGenerator(scanListPath=self.trainSplitPath, ImageRootPath=self.imageRootPath,
                                             Vocaubulary=self.vocabulary,depth=False)
        self.testDataset = DatasetGenerator(scanListPath=self.testSplitPath, ImageRootPath=self.imageRootPath,
                                            Vocaubulary=self.vocabulary, depth=False)
        self.trainDataLoader = DataLoader(dataset=self.trainDataset, shuffle=True, batch_size=self.batchSize,
                                          num_workers=8, pin_memory=self.pinMemory)
        self.testDataLoader = DataLoader(dataset=self.testDataset, shuffle=True, batch_size=self.batchSize,
                                         num_workers=8, pin_memory=self.pinMemory)
        print 'Dateset Loaded.'

        # Create model on multi-GPU
        self.rgbResNetModel = MyResNet_40()
        self.relevanceModel = RelevanceMap(wordEmbeddingNum=len(self.vocabulary),
                                           wordEmbeddingDim=self.wordEmbeddingDim)

        if torch.cuda.device_count() > 1:
            print("Use ", torch.cuda.device_count(), " GPUs!")

            self.rgbResNetModel = nn.DataParallel(self.rgbResNetModel)
            self.relevanceModel = nn.DataParallel(self.relevanceModel)

        self.rgbResNetModel = self.rgbResNetModel.cuda()
        self.relevanceModel = self.relevanceModel.cuda()

        print 'Model Created on GPUs.'

        self.rgbOptimizer = optim.Adam(self.rgbResNetModel.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
                                       weight_decay=self.weightDecay)
        self.relevanceOptimizer = optim.Adam(self.relevanceModel.parameters(), lr=self.lr, betas=(0.9, 0.999),
                                             eps=1e-08,
                                             weight_decay=self.weightDecay)

        self.rgbScheduler = ReduceLROnPlateau(self.rgbOptimizer, factor=0.1, patience=10, mode='min')
        self.relevanceScheduler = ReduceLROnPlateau(self.relevanceOptimizer, factor=0.1, patience=10, mode='min')

        self.loss = torch.nn.MSELoss()

        if self.checkPoint != "":
            self.load(self.checkPoint)


    def train(self):

        print 'Training task begin.'
        print '----Batch Size: %d' % self.batchSize
        print '----Learning Rate: %f' % self.lr
        print '----Epoch: %d' % self.epoch
        print '----Log Path: %s' % self.logPath
        print

        for self.epoch_i in range(self.epoch):

            self.epochTrain()
            self.test()

            if (self.epoch_i + 1) % self.saveEpoch == 0:
                self.save()

    def epochTrain(self):
        self.lossInEpoch = []
        s = '#' * 30 + '    Epoch %3d / %3d    ' % (self.epoch_i + 1, self.epoch) + '#' * 30
        print s

        for idx, (rgb, target, category) in enumerate(self.trainDataLoader):

            # Save some data for visualization
            try:
                if (idx + 1) % self.batchPerVisualize == 0 and self.visualize:
                    rgb_v = rgb[0, 0, :, :]
                    target_v = target[0,0,:,:]
                    category_v = self.vocabularyInv[category[0].data.numpy()[0]]
            except Exception,e:
                print e

            rgb = Variable(rgb).cuda()
            target = Variable(target, requires_grad=False).cuda()
            target = target.squeeze(1)
            category = Variable(category).cuda()

            feature = self.rgbResNetModel(rgb)
            releMap = self.relevanceModel(feature, category)
            lossValue = self.loss(input=releMap, target=target)

            self.lossInEpoch.append(lossValue.data)
            self.lossAll.append(lossValue.data)

            if (idx + 1) % self.batchPerlossDisplay == 0:
                print 'Epoch: %3d / %3d        Batch: %5d / %5d        BatchLoss: %6.4f        EpochAvgLoss: %6.4f' \
                      % (self.epoch_i + 1, self.epoch, idx + 1, np.ceil(len(self.trainDataset) / float(self.batchSize)),
                         self.lossInEpoch[-1], np.mean(self.lossInEpoch))

            self.rgbOptimizer.zero_grad()
            self.relevanceOptimizer.zero_grad()
            lossValue.backward()
            self.rgbOptimizer.step()
            self.relevanceOptimizer.step()

            if (idx + 1) % self.batchPerPlot == 0:
                self.plot()

            # Visualization
            try:
                if (idx + 1) % self.batchPerVisualize == 0 and self.visualize:
                    self.visualization(rgb_v,target_v,releMap[0,:,:],category_v,idx+1)
            except Exception,e:
                print e


        self.lossAllEpoch.append(np.mean(self.lossInEpoch))

    def test(self):
        self.testLossInEpoch = []
        self.successRateInEpoch = []
        s = '#' * 15 + '  Epoch %3d / %3d Test ' % (self.epoch_i + 1, self.epoch) + '#' * 15
        print s

        success_rate_list = []
        suceess_all_rate = 0 
        for idx, (rgb, target, category) in enumerate(self.testDataLoader):
            rgb = Variable(rgb).cuda()
            target = Variable(target, requires_grad=False).cuda()
            category = Variable(category).cuda()

            feature = self.rgbResNetModel(rgb)
            releMap = self.relevanceModel(feature, category)
            lossValue = self.loss(input=releMap, target=target)
            success_rate = self.success_rate(releMap.detach(),target.detach()
            self.successRateInEpoch.append(success_rate)
            self.lossInEpoch.append(lossValue.item())
            self.lossAll.append(lossValue.item())
            
            self.testLossInEpoch.append(lossValue.item())
        

        self.testSuccessAllEpoch.append(np.mean(self.successRateInEpoch))
        self.testLossAllEpoch.append(np.mean(self.testLossInEpoch))
        print 'Epoch: %3d / %3d        Test Loss: %6.2f        Success Rate: %6.2f' % (self.epoch_i + 1, self.epoch, self.testLossAllEpoch[-1], self.testSuccessAllEpoch[-1])

    def save(self):
        path = self.logPath + '/Epoch-%03d.pth.tar' % (self.epoch_i + 1)
        torch.save({'epochs': self.epoch_i + 1,
                    'batch_size': self.batchSize,
                    'learning_rate': self.lr,
                    'rgbResnet_model_state_dict': self.rgbResNetModel.state_dict(),
                    'releMap_model_state_dict': self.relevanceModel.state_dict(),
                    'training_loss': self.lossAll,
                    'training_loss_epoch': self.lossAllEpoch,
                    'test_loss': self.testLossAllEpoch,
                    'test_loss_best': np.max(self.testLossAllEpoch),
                    'rgbResnet_optimizer': self.rgbOptimizer.state_dict(),
                    'releMap_optimizer': self.relevanceOptimizer.state_dict()},
                   path)
        print 'Training log saved to %s' % path

    def load(self, path):
        print 'Load model from: ' + path

        modelCheckpoint = torch.load(path)
        self.rgbResNetModel.load_state_dict(modelCheckpoint['rgbResnet_model_state_dict'])
        self.relevanceModel.load_state_dict(modelCheckpoint['releMap_model_state_dict'])
        self.rgbOptimizer.load_state_dict(modelCheckpoint['rgbResnet_optimizer'])
        self.relevanceOptimizer.load_state_dict(modelCheckpoint['releMap_optimizer'])

        self.lossAll = modelCheckpoint['training_loss']
        self.lossAllEpoch = modelCheckpoint['training_loss_epoch']
        self.testLossAllEpoch = modelCheckpoint['test_loss']

        return modelCheckpoint

    def plot(self):
        plt.switch_backend('agg')
        plt.figure(figsize=(24, 12))

        plt.subplot(2, 2, 1)
        plt.plot(self.lossInEpoch)
        plt.title('Training Loss of Current Epoch')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        # plt.ylim([0, 1.2])
        plt.xlim([0, self.batchSize])
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
        plt.xlim([0, max(len(self.lossAllEpoch), 1)])
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

    def visualization(self,rgb_v,target_v,releMap,category_v,batchIdx):
        plt.switch_backend('agg')
        plt.figure(figsize=(24, 24), dpi=100)

        plt.figure()
        plt.suptitle(category_v, fontsize=16)
        plt.subplot(1,3,1)
        plt.imshow(rgb_v.data.numpy())

        plt.subplot(1,3,2)
        plt.imshow(target_v.data.numpy())

        plt.subplot(1,3,3)
        plt.imshow(releMap.detach().cpu().data.numpy())

        dirPath = os.path.join(self.logPath, 'visualization')
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)

        figPath = os.path.join(dirPath, 'Epoch%03d-Batch%05d.png' % (self.epoch_i + 1, batchIdx))
        plt.savefig(figPath)

    def success_rate(self, releMap, target):
        # return the rate for success aline
        threshold = 0.7
        releMap = releMap.squeeze()
        target = target.squeeze()
        success_rate = 0
        for i in range(self.batchSize):
            count = 0
            success_num = 0
            for x, row in enumerate(releMap[i]):
                for y, value in enumerate(row):
                    count += 1
                    if value > threshold:
                        releMap[i][x][y]=1
                    else:
                        releMap[i][x][y]=0
                    if releMap[i][x][y]==target[i][x][y]:
                        success_num += 1
            success_rate += float(success_num)/float(count)
        success_rate = success_rate / self.batchSize
        return success_rate
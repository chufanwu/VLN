import os

import torch
from torchvision import transforms
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from goalmap_model import MyResNet_40, RelevanceMap, MyResNet101_FasterRCNN
from goalmap_utils import buildVocaulary, generateTopView
from collections import OrderedDict


class Model():
    def __init__(self, modelPath):
        objectCategoryPath = './data/category_mapping.tsv'
        regionCategoryPath = './data/region_category.txt'

        self.objectCategoryPath = './data/category_mapping.tsv'
        self.regionCategoryPath = './data/region_category.txt'
        self.trainSplitPath = './dataset_split/train_scans.txt'
        self.testSplitPath = './dataset_split/test_scans.txt'
        self.imageRootPath = '/home/qiyuand/matterport3D'

        self.vocabulary, self.vocabularyInv = buildVocaulary(regionCategoryPath, objectCategoryPath)

        self.wordEmbeddingDim = 256
        self.rgbResNetModel = MyResNet_40(dropout=False)
        self.relevanceModel = RelevanceMap(wordEmbeddingNum=len(self.vocabulary),
                                           wordEmbeddingDim=self.wordEmbeddingDim,
                                           dropout=False)

        transformList = []
        transformList.append(transforms.ToTensor())
        transformList.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.rgbTransformSequence = transforms.Compose(transformList)

        transformList = []
        transformList.append(transforms.ToTensor())
        # transformList.append(transforms.Resize((40,40)))
        self.targetTransformSequence = transforms.Compose(transformList)

        self.rgbResNetModel = self.rgbResNetModel.cuda()
        self.relevanceModel = self.relevanceModel.cuda()

        self.load(modelPath)

        self.rgbResNetModel.eval()
        self.relevanceModel.eval()


    def display(self, scan, imageId, category):
        assert self.vocabulary.has_key(category)
        category_ori = category
        rgbPath = os.path.join(self.imageRootPath, 'sampled_color_images', scan, imageId + '.jpg')
        depthPath = os.path.join(self.imageRootPath, 'sampled_depth_images', scan, imageId + '.png')
        targetPath = os.path.join(self.imageRootPath, 'generated_semantic_training_images', scan, imageId+'_'+\
                                  category + '.png')

        if  os.path.exists(targetPath):
            targetImage = plt.imread(targetPath)
        else:
            targetImage = np.zeros((640,640))

        rgbImage = Image.open(rgbPath)
        rgb = self.rgbTransformSequence(rgbImage)

        rgb = torch.reshape(rgb,(1,3,640,640))
        rgb = Variable(rgb).cuda()
        category = Variable(torch.LongTensor([self.vocabulary[category]])).cuda()
        category = torch.reshape(category,(1,1))

        # Go through the models
        feature = self.rgbResNetModel(rgb)
        releMap = self.relevanceModel(feature, category)

        rgbImage = plt.imread(rgbPath)
        depthImage = plt.imread(depthPath)

        releMap = torch.reshape(releMap,(1,1,40,40)).detach().cpu().numpy()
        releMap = releMap[0,0,:,:]
        releMap = cv2.resize(releMap,(0,0),fx=16,fy=16)
        releMap = releMap.reshape(1,1,640,640)

        rgb = cv2.resize(rgbImage,(0,0),fx=1,fy=1)
        rgb = np.transpose(rgb,(2,0,1))
        rgb = rgb.reshape(1,3,640,640)
        rgb = (rgb / 255.0).astype('float32')
        rgbTarget = np.concatenate((rgb,releMap),axis=1)
        rgbTarget = Variable(torch.from_numpy(rgbTarget)).cuda()
        depthImage = cv2.resize(depthImage,(0,0),fx=1,fy=1)
        depthImage = depthImage.reshape(1,1,640,640)

        topView = generateTopView(rgbTarget, depthImage,size=100)
        topView = topView.cpu().numpy()
        rgbTopView = topView[0,0:3,:,:]
        rgbTopView = (np.transpose(rgbTopView,(1,2,0)) * 255.0).astype('uint8')
        releMapTopView = topView[0,3,:,:]

        plt.switch_backend('agg')
        plt.figure(figsize=(35, 35), dpi=150)

        plt.figure()
        plt.suptitle(category_ori, fontsize=16)

        plt.subplot(2, 3, 1)
        plt.imshow(rgbImage)

        plt.subplot(2, 3, 2)
        plt.imshow(targetImage)

        plt.subplot(2, 3, 3)
        plt.imshow(releMap[0,0,:,:])
        #
        # rgb_rele = (cv2.resize(rgbImage,(0,0),fx=(1.0/16),fy=(1.0/16))).astype('uint8')
        # plt.subplot(2, 3, 4)
        # plt.imshow(rgb_rele,alpha=0.5)
        # plt.imshow((releMap[0,0,:,:] * 255).astype('uint8'),alpha=0.5)


        plt.subplot(2, 3, 5)
        plt.imshow(rgbTopView)

        plt.subplot(2, 3, 6)
        plt.imshow(releMapTopView)

        plt.savefig('./cache/cache.png')



    def load(self, path):
        print 'Load model from: ' + path

        modelCheckpoint = torch.load(path)
        # print modelCheckpoint.keys()
        # print modelCheckpoint['rgbResnet_model_state_dict'].keys()
        # self.rgbResNetModel.load_state_dict(modelCheckpoint['rgbResnet_model_state_dict'])
        # self.relevanceModel.load_state_dict(modelCheckpoint['releMap_model_state_dict'])



        rgb_state_dict = OrderedDict()
        for k, v in modelCheckpoint['rgbResnet_model_state_dict'].items():
            name = k[7:]  # remove `module.`
            rgb_state_dict[name] = v
        # load params
        self.rgbResNetModel.load_state_dict(rgb_state_dict)

        rele_state_dict = OrderedDict()
        for k, v in modelCheckpoint['releMap_model_state_dict'].items():
            name = k[7:]  # remove `module.`
            rele_state_dict[name] = v
        # load params
        self.relevanceModel.load_state_dict(rele_state_dict)

        return modelCheckpoint


class Model_FasterRCNN():
    def __init__(self, modelPath):
        objectCategoryPath = './data/category_mapping.tsv'
        regionCategoryPath = './data/region_category.txt'

        self.objectCategoryPath = './data/category_mapping.tsv'
        self.regionCategoryPath = './data/region_category.txt'
        self.trainSplitPath = './dataset_split/train_scans.txt'
        self.testSplitPath = './dataset_split/test_scans.txt'
        self.imageRootPath = '/home/qiyuand/matterport3D'

        self.vocabulary, self.vocabularyInv = buildVocaulary(regionCategoryPath, objectCategoryPath)

        self.wordEmbeddingDim = 256
        self.rgbResNetModel = MyResNet101_FasterRCNN(dropout=False)
        self.relevanceModel = RelevanceMap(wordEmbeddingNum=len(self.vocabulary),
                                           wordEmbeddingDim=self.wordEmbeddingDim,
                                           dropout=False,
                                           featureChannel=1024)

        transformList = []
        transformList.append(transforms.Resize((448,448)))
        transformList.append(transforms.ToTensor())
        transformList.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.rgbTransformSequence = transforms.Compose(transformList)

        transformList = []
        transformList.append(transforms.ToTensor())
        # transformList.append(transforms.Resize((40,40)))
        self.targetTransformSequence = transforms.Compose(transformList)

        self.rgbResNetModel = self.rgbResNetModel.cuda()
        self.relevanceModel = self.relevanceModel.cuda()

        self.load(modelPath)

        self.rgbResNetModel.eval()
        self.relevanceModel.eval()


    def display(self, scan, imageId, category):
        assert self.vocabulary.has_key(category)
        category_ori = category
        rgbPath = os.path.join(self.imageRootPath, 'sampled_color_images', scan, imageId + '.jpg')
        depthPath = os.path.join(self.imageRootPath, 'sampled_depth_images', scan, imageId + '.png')
        targetPath = os.path.join(self.imageRootPath, 'generated_semantic_training_images', scan, imageId+'_'+\
                                  category + '.png')

        if  os.path.exists(targetPath):
            targetImage = plt.imread(targetPath)
        else:
            targetImage = np.zeros((448,448))

        rgbImage = Image.open(rgbPath)
        rgb = self.rgbTransformSequence(rgbImage)

        rgb = torch.reshape(rgb,(1,3,448,448))
        rgb = Variable(rgb).cuda()
        category = Variable(torch.LongTensor([self.vocabulary[category]])).cuda()
        category = torch.reshape(category,(1,1))

        # Go through the models
        feature = self.rgbResNetModel(rgb)
        releMap = self.relevanceModel(feature, category)

        rgbImage = plt.imread(rgbPath)
        depthImage = plt.imread(depthPath)

        releMap = torch.reshape(releMap,(1,1,28,28)).detach().cpu().numpy()
        releMap = releMap[0,0,:,:]
        releMap = cv2.resize(releMap,(0,0),fx=16,fy=16)
        releMap = releMap.reshape(1,1,448,448)

        rgb = cv2.resize(rgbImage,(0,0),fx=0.7,fy=0.7)
        rgb = np.transpose(rgb,(2,0,1))
        rgb = rgb.reshape(1,3,448,448)
        rgb = (rgb / 255.0).astype('float32')
        rgbTarget = np.concatenate((rgb,releMap),axis=1)
        rgbTarget = Variable(torch.from_numpy(rgbTarget)).cuda()
        depthImage = cv2.resize(depthImage,(0,0),fx=0.7,fy=0.7)
        depthImage = depthImage.reshape(1,1,448,448)

        topView = generateTopView(rgbTarget, depthImage,size=100)
        topView = topView.cpu().numpy()
        rgbTopView = topView[0,0:3,:,:]
        rgbTopView = (np.transpose(rgbTopView,(1,2,0)) * 255.0).astype('uint8')
        releMapTopView = topView[0,3,:,:]

        plt.switch_backend('agg')
        plt.figure(figsize=(35, 35), dpi=150)

        plt.figure()
        plt.suptitle(category_ori, fontsize=16)

        plt.subplot(2, 3, 1)
        plt.imshow(rgbImage)

        plt.subplot(2, 3, 2)
        plt.imshow(targetImage)

        plt.subplot(2, 3, 3)
        plt.imshow(releMap[0,0,:,:])
        #
        # rgb_rele = (cv2.resize(rgbImage,(0,0),fx=(1.0/16),fy=(1.0/16))).astype('uint8')
        # plt.subplot(2, 3, 4)
        # plt.imshow(rgb_rele,alpha=0.5)
        # plt.imshow((releMap[0,0,:,:] * 255).astype('uint8'),alpha=0.5)


        plt.subplot(2, 3, 5)
        plt.imshow(rgbTopView)

        plt.subplot(2, 3, 6)
        plt.imshow(releMapTopView)

        plt.savefig('./cache/cache.png')



    def load(self, path):
        print 'Load model from: ' + path

        modelCheckpoint = torch.load(path)
        # print modelCheckpoint.keys()
        # print modelCheckpoint['rgbResnet_model_state_dict'].keys()
        # self.rgbResNetModel.load_state_dict(modelCheckpoint['rgbResnet_model_state_dict'])
        # self.relevanceModel.load_state_dict(modelCheckpoint['releMap_model_state_dict'])



        rgb_state_dict = OrderedDict()
        for k, v in modelCheckpoint['rgbResnet_model_state_dict'].items():
            name = k[7:]  # remove `module.`
            rgb_state_dict[name] = v
        # load params
        self.rgbResNetModel.load_state_dict(rgb_state_dict)

        rele_state_dict = OrderedDict()
        for k, v in modelCheckpoint['releMap_model_state_dict'].items():
            name = k[7:]  # remove `module.`
            rele_state_dict[name] = v
        # load params
        self.relevanceModel.load_state_dict(rele_state_dict)

        return modelCheckpoint



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    modelPath = '/home/qiyuand/SemanticMap/log/2018-08-21-13-48-47/models/Epoch-001-Batch-2800.pth.tar'
    scan = '1LXtFkjw3qL'
    imageId = '14a8edbbe4b14a05b1b5782a884fb6bf_12'

    model = Model_FasterRCNN(modelPath)
    model.display(scan=scan, imageId=imageId, category='wall')


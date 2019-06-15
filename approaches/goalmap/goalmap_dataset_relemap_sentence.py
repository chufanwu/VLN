import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from goalmap_utils import buildVocaulary


class DatasetGenerator(Dataset):
    def __init__(self, scanListPath, ImageRootPath, Vocaubulary, depth):

        self.ImageRootPath = ImageRootPath
        self.rgbImage = []
        self.depthImage = []
        self.targetImage = []
        self.category = []
        self.depth = depth
        self.generatedImagePath = 'generated_semantic_training_images_2'
        self.sentencePatternPath = './data/sentence_pattern.txt'

        objectCategoryPath = './data/category_mapping.tsv'
        regionCategoryPath = './data/region_category.txt'
        self.vocabulary, _ =  buildVocaulary(regionCategoryPath, objectCategoryPath)

        self.sentencePattern = []
        with open(self.sentencePatternPath,'r') as fp:
            for line in fp:
                if '\n' in line:
                    line = line[:-1]
                self.sentencePattern.append(line)
        self.sentencePatternLen = len(self.sentencePattern)

        transformList = []
        transformList.append(transforms.Resize((448,448)))
        transformList.append(transforms.ToTensor())
        transformList.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.rgbTransformSequence = transforms.Compose(transformList)

        transformList = []
        transformList.append(transforms.ToTensor())
        self.targetTransformSequence = transforms.Compose(transformList)



        scanListPath = scanListPath
        scans = []
        with open(scanListPath, 'r') as fp:
            for line in fp:
                if line[-1] == '\n':
                    line = line[:-1]
                scans.append(line)

        for scan in scans:
            scanTrainingPath = os.path.join(ImageRootPath, self.generatedImagePath, scan)
            items = [i for i in os.listdir(scanTrainingPath) if len(i) > 10]
            for item in items:
                imageId = item.split('_')[0] + '_' + item.split('_')[1]
                category = item.split('_')[-1][:-4]
                rgbPath = os.path.join(ImageRootPath, 'sampled_color_images', scan, imageId + '.jpg')
                depthPath = os.path.join(ImageRootPath, 'sampled_depth_images', scan, imageId + '.png')
                targetPath = os.path.join(ImageRootPath, self.generatedImagePath, scan, item)

                if category not in ['floor', 'wall', 'ceiling'] or (
                        category in ['floor', 'wall', 'ceiling'] and np.random.rand() <= 0.1):
                    if category not in ['void', 'hallway', 'kitchen','library','balcony'] and ('room' not in category):
                        if os.path.exists(rgbPath) and os.path.exists(depthPath) and os.path.exists(targetPath):
                            self.rgbImage.append(rgbPath)
                            self.depthImage.append(depthPath)
                            self.targetImage.append(targetPath)
                            self.category.append(category)

    def __getitem__(self, item):

        if self.depth:
            rgbImage = plt.imread(self.rgbImage[item])
            depthImage = plt.imread(self.depthImage[item]).reshape(1, 640, 640).astype('float32')
            targetImage = plt.imread(self.targetImage[item]).reshape(1, 640, 640).astype('float32')
            category = self.category[item]

            rgbImage = self.rgbTransformSequence(rgbImage)
            depthImage = torch.from_numpy(depthImage)
            targetImage = torch.from_numpy(targetImage)

            return rgbImage, depthImage, targetImage, self.vocabulary[category]

        else:

            rgbImage = Image.open(self.rgbImage[item])
            targetImage = Image.open(self.targetImage[item])
            targetImage = targetImage.resize((28,28))
            category = self.category[item]

            patternId = np.random.choice(self.sentencePatternLen,1).tolist()[0]
            sentence = self.sentencePattern[patternId] % category

            rgbImage = self.rgbTransformSequence(rgbImage)
            targetImage = self.targetTransformSequence(targetImage)

            return rgbImage, targetImage, sentence

    def __len__(self):
        return len(self.rgbImage)


if __name__ == '__main__':
    splitPath = './dataset_split/tmp.txt'
    ImageRootPath = '/Volumes/Dongqiyuan/matterport_dataset'

    objectCategoryPath = './data/category_mapping.tsv'
    regionCategoryPath = './data/region_category.txt'

    vocabulary, vocabularyInv = buildVocaulary(regionCategoryPath, objectCategoryPath)

    tmpDataset = DatasetGenerator(scanListPath=splitPath, ImageRootPath=ImageRootPath,Vocaubulary=vocabulary,depth=False)
    tmpDataLoader = DataLoader(dataset=tmpDataset, batch_size=4, shuffle=True, num_workers=4)

    for idx,(rgb,target,instruction) in enumerate(tmpDataLoader):
        pass

#!/usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
from glob import glob
import argparse
from helpers2    import *
from matplotlib import pyplot as plt

class BOV:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.im_helper = ImageHelpers()
        self.bov_helper = BOVHelpers(no_clusters)
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []

    def trainModel(self,trainset):
        print "\nTrain Mode"
        print "Training for trainset =", trainset
        stage = 'Train/'
        classes = os.listdir('caltech101/')
        #classes = classes[51:51+(len(classes)/50)]
        # read file. prepare file lists.
        count = 0
        self.images = []
        train_size = []
        for i in xrange(len(classes)):
            im, count, size= self.file_helper.getFiles(classes[i],stage,trainset)
            self.trainImageCount+=count
            self.images+=im
            train_size.append(size)
        # extract SIFT Features from each image
        i = 0
        for label_count in xrange(0, len(classes)):
            self.name_dict[str(label_count)] = classes[label_count]
            print "Computing Features for ", classes[label_count] , "-", label_count
            images = self.images[i:i+trainset]
            i+=train_size[label_count]
            for j in xrange(len(images)):
                self.train_labels = np.append(self.train_labels,label_count)
                kp, des = self.im_helper.features(images[j])
                self.descriptor_list.append(des)
            label_count += 1

        del train_size
        del self.images

        print "\nClustering Features\n"
        bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)

        print "Using Kmeans to Perform Clustering\n"
        self.bov_helper.cluster()

        print "Preparing the Vocabulary\n"
        self.bov_helper.developVocabulary(n_images = self.trainImageCount, descriptor_list=self.descriptor_list)

        # show vocabulary trained
        #self.bov_helper.plotHist()

        print "Data Normalization\n"
        self.bov_helper.standardize()
        self.bov_helper.train(self.train_labels)
        self.saveDictionary(trainset)

    def recognize(self,test_img):
        # Taking vocabulary from test image
        kp, des = self.im_helper.features(test_img) # keypoints, descriptors
        test_ret = self.bov_helper.kmeans_obj.predict(des)
        vocab = np.array( [[ float(0) for i in range(self.no_clusters)]])
        for each in test_ret:
            vocab[0][each] += 1.0
        # Scale the features
        vocab = self.bov_helper.scale.transform(vocab)
        # predict the class of the image
        lb = self.bov_helper.clf.predict(vocab)
        #print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
        return lb

    def testModel(self, foundtxt,trainset):
        print "\nTest Mode\n\n"
        print "Testing for trainset =", trainset
        if foundtxt == 1:
            self.loadDictionary(trainset)
        stage = 'Test/'
        classes = os.listdir('caltech101')
        #classes = classes[0:(len(classes))]
        testImageCount = 0
        self.images = []
        test_size = []
        for i in xrange(len(classes)):
            im, count, size = self.file_helper.getFiles(classes[i],stage,trainset)
            testImageCount+=count
            self.images+=im
            test_size.append(size)
        # extract SIFT Features from each image
        predictions = []
        true = 0
        true_class = 0
        trueClass = []
        i = 0
        for label_count in xrange(0,len(classes)):
            print "processing " , classes[label_count]
            images = self.images[i:i+test_size[label_count]]
            i+=test_size[label_count]
            if classes[label_count] != 'elephant':
                if label_count != 89:
                    for j in xrange(len(images)):
                        cl = self.recognize(images[j])
                        predictions.append({
                            'image' :images[j],
                            'class':cl,
                            'object_name':self.name_dict[str(int(cl[0]))]
                            })
                        if(self.name_dict[str(int(cl[0]))] == classes[label_count]):
                            true+=1
                            true_class+=1.0
            trueClass.append(true_class)
            true_class = 0
            label_count += 1
        cv2.destroyAllWindows()
        print "Training Set = ",trainset
        print "Number of positives = ", float(true)/float(len(self.images))
        for i in xrange(len(trueClass)):
            print classes[i]
            print trueClass[i]/test_size[i]
        # Garbage collector
        del self.images
        del test_size
        del classes
        del trueClass

    def saveDictionary(self,trainset):
        data = np.array([self.name_dict,self.bov_helper.clf,self.bov_helper.kmeans_obj, self.bov_helper.scale]) # Dictionary, SVM model, Kmeans model,
        with open('dictionary{}.txt'.format(trainset),'w') as outfile:
            outfile.write(data.dumps())

    def loadDictionary(self,trainset):
        with open('dictionary{}.txt'.format(trainset),'r') as infile:
            data = np.loads(infile.read())
        self.name_dict = data[0]
        self.bov_helper.clf = data[1]
        self.bov_helper.kmeans_obj = data[2]
        self.bov_helper.scale = data[3]

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    print "################################################################"
    print "                 Bag of Visual Words                            "
    print "            Train set = 10,15,20,25 and 30                      "
    print "              Opencv 3.4.1 -- python 2.7                        "
    print "################################################################"
    print "\n\n"
    bov = BOV(no_clusters=20)
    for train_set in xrange(10,35,5):
        foundtxt = 0
        model = os.listdir('.')
        for i in xrange(len(model)):
            if(model[i] == 'dictionary{}.txt'.format(train_set)):
                foundtxt = 1
        if(foundtxt == 0):
            # train the model
            bov.trainModel(train_set)
        # test model
        bov.testModel(foundtxt,train_set)

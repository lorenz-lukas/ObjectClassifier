#!/usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
from glob import glob
import argparse
from helpers import *
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

    def trainModel(self):
        stage = 'Train/'
        classes = os.listdir('CalTech101/Train/')
        #classes = classes[51:51+(len(classes)/50)] #Select less classes
        # read file. prepare file lists.
        self.images = []
        for i in xrange(len(classes)):
            im, count = self.file_helper.getFiles(classes[i],stage)
            self.trainImageCount+=count
            self.images+=im
        # extract SIFT Features from each image
        label_count = 0
        for i in xrange(0,self.trainImageCount,5):
            self.name_dict[str(label_count)] = classes[label_count]
            print "Computing Features for ", classes[label_count] , "-", label_count
            images = self.images[i:i+5]
            for j in xrange(len(images)):
                self.train_labels = np.append(self.train_labels,label_count)
                kp, des = self.im_helper.features(images[j])
                self.descriptor_list.append(des)
            label_count += 1

        print "\nClustering Features\n"
        bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)

        print "Using Kmeans to Perform Clustering\n"
        self.bov_helper.cluster()

        print "Preparing the Vocabulary\n"
        self.bov_helper.developVocabulary(n_images = self.trainImageCount, descriptor_list=self.descriptor_list)

        # show vocabulary trained
        self.bov_helper.plotHist()

        print "Data Normalization\n"
        self.bov_helper.standardize()
        self.bov_helper.train(self.train_labels)
        self.saveDictionary()

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
        print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
        return lb

    def testModel(self, foundtxt):
        if foundtxt == 1:
            self.loadDictionary()
        stage = 'Test/'
        classes = os.listdir('CalTech101/Test/')
        #classes = classes[51:51+(len(classes)/50)]
        testImageCount = 0
        self.images = []
        for i in xrange(len(classes)):
            im, count = self.file_helper.getFiles(classes[i],stage)
            testImageCount+=count
            self.images+=im
        # extract SIFT Features from each image
        label_count = 0
        predictions = []
        true = 0
        im_list = 20
        true_class = 0
        trueClass = []
        classes_size = []
        for i in xrange(0,testImageCount,im_list):
            print "processing " , classes[label_count]
            (db, Folders, im_list) = os.walk('CalTech101/'+stage+classes[label_count]+'/').next()
            images = self.images[i:i+len(im_list)]
            classes_size.append(float(len(im_list)))
            for j in xrange(len(images)):
                if classes[label_count] != 'elephant' and j != 18:
                    #cv2.imshow("im",images[j])
                    #cv2.waitKey(100)
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
        print "Number of positives = ", float(true)/float(len(self.images))
        for i in xrange(len(trueClass)):
            print trueClass[i]/classes_size[i]
        #for each in predictions:
        #    cv2.imshow(each['object_name'], each['image'])
        #    cv2.waitKey()
        #    cv2.destroyWindow(each['object_name'])
        #    plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
        #    plt.title(each['object_name'])
        #    plt.show()
    def print_vars(self):
        pass

    def saveDictionary(self):
        data = np.array([self.name_dict,self.bov_helper.clf,self.bov_helper.kmeans_obj, self.bov_helper.scale]) # Dictionary, SVM model, Kmeans model,
        with open('dictionary2.txt','w') as outfile:
            outfile.write(data.dumps())

    def loadDictionary(self):
        with open('dictionary2.txt','r') as infile:
            data = np.loads(infile.read())
        self.name_dict = data[0]
        self.bov_helper.clf = data[1]
        self.bov_helper.kmeans_obj = data[2]
        self.bov_helper.scale = data[3]

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    print "################################################################"
    print "                 Bag of Visual Words                            "
    print "           Opencv 3.4.1 -- python 2.7                           "
    print "################################################################"
    print "\n\n"
    foundtxt = 0
    model = os.listdir('.')
    for i in xrange(len(model)):
        if(model[i] == 'dictionary2.txt'):
            foundtxt = 1
    #if(foundtxt == 0):
        # train the model
    bov.trainModel()
    # test model
    bov.testModel(foundtxt)

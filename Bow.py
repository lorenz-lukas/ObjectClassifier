#!/usr/bin/env python
# coding=utf-8
import cv2
import numpy as np
from glob import glob
import os
from matplotlib import pyplot as plt
sift_object = cv2.xfeatures2d.SIFT_create()
def train(directory,classes):
    global sift_object
    imlist = []
    name_dic = {}
    train_labels = np.array([])
    descriptor_list = []
    # Getting images
    for c in xrange(len(classes)):
        (db, Folders, img_list) = os.walk(directory+classes[c]+'/').next()
        for i in xrange(len(img_list)):
            img = cv2.imread(directory+classes[c]+'/'+img_list[i])
            imlist.append(img)
    # Extracting features with SIFT
    counter = 0
    for label_count in xrange(0,len(imlist),len(img_list)):
        name_dic[str(counter)] = classes[counter]
        print "Computing Features for ", classes[counter]
        images = imlist[label_count:label_count+5]
        for i in xrange(len(images)):
            cv2.imshow("im", images[i])
            cv2.waitKey(20)
            train_labels = np.append(train_labels, label_count)
            kp, des = sift_object.detectAndCompute(images[i], None)
            descriptor_list.append(des)
        counter+=1
    # perform clustering
    bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)
    self.bov_helper.cluster()
    self.bov_helper.developVocabulary(n_images = self.trainImageCount, descriptor_list=self.descriptor_list)

    # show vocabulary trained
    # self.bov_helper.plotHist()


    self.bov_helper.standardize()
    self.bov_helper.train(self.train_labels)

def main():
    directory_train = 'CalTech101/Train/'
    directory_test = 'CalTech101/Test/'
    classes_train = os.listdir(directory_train)
    train(directory = directory_train , classes = classes_train)

main()

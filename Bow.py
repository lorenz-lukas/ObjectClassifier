#!/usr/bin/env python
# coding=utf-8
import cv2
import numpy as np
import os
import json
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import ast

sift_object = cv2.xfeatures2d.SIFT_create()

s = SVC()
k = KMeans(n_clusters = 20)

def bow(directory,classes,no_clusters):
    global sift_object,s,k
    imlist = []
    name_dic = {}
    train_labels = np.array([])
    descriptor_list = []
    descriptor_vstack = None
    std = None
    # Getting images
    for c in xrange(len(classes)):
        (db, Folders, img_list) = os.walk(directory+classes[c]+'/').next()
        for i in xrange(len(img_list)):
            img = cv2.imread(directory+classes[c]+'/'+img_list[i])
            imlist.append(img)
    # Extracting features with SIFT
    print "Computing features with SIFT\n\n"
    counter = 0
    for label_count in xrange(0,len(imlist),len(img_list)):
        name_dic[str(counter)] = classes[counter]
        print "Computing Features for ", classes[counter]
        images = imlist[label_count:label_count+5]
        for i in xrange(len(images)):
            #cv2.imshow("im", images[i])
            #cv2.waitKey(20)
            train_labels = np.append(train_labels, label_count)
            kp, des = sift_object.detectAndCompute(images[i], None)
            descriptor_list.append(des)
        counter+=1
    print "Total = ", len(classes)
    cv2.destroyAllWindows()
    # perform clustering
    print "\nClustering Features\n"
    bov_descriptor_stack = formatND(descriptor_list)
    print "Using Kmeans to Perform Clustering\n"
    kmeans_ret = k.fit_predict(bov_descriptor_stack)
    print "Preparing the Vocabulary\n"
    vocab = developVocabulary(n_clusters = no_clusters, n_images = len(imlist), descriptor_list = descriptor_list, kmeans_model = kmeans_ret)
    print "Data Normalization\n"
    # show vocabulary trained
    #plotHist(vocabulary = vocab, n_clusters = no_clusters)
    # Data normalization
    if std is None:
        scale = StandardScaler().fit(vocab)
        vocab = scale.transform(vocab)
    else:
        print "STD not none. External STD supplied"
        vocab = std.transform(vocab)
    model = train(vocabulary = vocab, labels = train_labels)
    saveDictionary(m = model,v = vocab,dic = name_dic)
    return model, vocab, name_dic

def formatND(l):
    vStack = np.array(l[0])
    for remaining in l[1:]:
        vStack = np.vstack((vStack, remaining))
        descriptor_vstack = vStack.copy()
    return descriptor_vstack

def developVocabulary(n_clusters,n_images, descriptor_list, kmeans_model ):
    mega_histogram = np.array([np.zeros(n_clusters) for i in range(n_images)])
    old_count = 0
    for i in range(n_images):
        l = len(descriptor_list[i])
        for j in range(l):
            if kmeans_model is None:
                idx = kmeans_model[old_count+j]
            else:
                idx = kmeans_model[old_count+j]
            mega_histogram[i][idx] += 1
        old_count += l
    print "Vocabulary Histogram Generated"
    return mega_histogram

def plotHist(vocabulary, n_clusters):
    print "Plotting histogram"
    x_scalar = np.arange(n_clusters)
    y_scalar = np.array([abs(np.sum(vocabulary[:,h], dtype=np.int32)) for h in range(n_clusters)])

    print y_scalar

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()

def train(vocabulary, labels):
    global s
    print "Training SVM"
    print s
    #print "Train labels", labels
    model = s.fit(vocabulary, labels)
    print "Training completed\n", model
    return model

def saveDictionary(m,v,dic):
    global k
    data = np.array([m,v,dic,k])
    with open('dictionary.txt','w') as outfile:
        outfile.write(data.dumps())

############################### TEST METHODS ##################################

def loadDictionary():
    global k
    with open('dictionary.txt','r') as infile:
        data = np.loads(infile.read())
    return data[0], data[1], data[2], data[3]

def bow_model():
    global k
    m,v,dic,k = loadDictionary()
    #print "\nClustering Features\n"
    #bov_descriptor_stack = formatND(d)
    #print "Using Kmeans to Perform Clustering\n"
    #kmeans_ret = k.fit_predict(bov_descriptor_stack)
    #model = train(vocabulary = v, labels = l)
    return m,v,dic

def test(directory, classes, bw, no_clusters):
    print "\n\n\nTesting BOW model\n\n\n"
    imlist = []
    predictions = []
    for c in xrange(len(classes)):
        (db, Folders, img_list) = os.walk(directory+classes[c]+'/').next()
        for i in xrange(len(img_list)):
            img = cv2.imread(directory+classes[c]+'/'+img_list[i])
            imlist.append(img)
    name_dic = bw[2]
    for im in imlist:
        print "processing " ,classes[i]
        #for im in imglist:
        # print imlist[0].shape, imlist[1].shape
        #print im.shape
        cv2.imshow("im",im)
        cv2.waitKey(300)
        cl,vocab = recognize(im, bw[0], no_clusters,bw[2])
        predictions.append({
            'image':im,
            'class':cl,
            'object_name':name_dic[str(int(cl[0]))]
            })
    print predictions
    cv2.destroyAllWindows()
    for each in predictions:
        # cv2.imshow(each['object_name'], each['image'])
        # cv2.waitKey()
        # cv2.destroyWindow(each['object_name'])
        plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
        plt.title(each['object_name'])
        plt.show()

def recognize(test_img, model, no_clusters,name_dict):
    global s,sift_object,k
    kp, des = sift_object.detectAndCompute(test_img, None)
    # generate vocab for test image
    vocab = np.array( [[ 0 for i in range(no_clusters)]])
    # locate nearest clusters for each of
    test_ret = k.predict(des)
    for each in test_ret:
        vocab[0][each] += 1
    print "Vocabulary"
    #print vocab
    # Scale the features
    #vocab = transform(float(vocab))
    scale = StandardScaler().fit(vocab)
    print scale
    vocab = scale.transform(vocab)
    #print vocab
    # predict the class of the image
    lb = model.predict(vocab)
    print lb
    #print "Image belongs to class : ", name_dict[str(int(lb[0]))]
    return lb,vocab

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print "################################################################"
    print "                 Bag of Visual Words                            "
    print "           Opencv 3.4.1 -- python 2.7                           "
    print "################################################################"
    print "\n\n"
    t = 0
    directory_train = 'CalTech101/Train/'
    directory_test = 'CalTech101/Test/'  # Same number of train classes
    model = os.listdir('.')
    classes_train = os.listdir(directory_train)
    for i in xrange(len(model)):
        if(model[i] == 'dictionary.txt'):
            t = 1
    if(t == 0):
        bag = bow(directory = directory_train , classes = classes_train, no_clusters = 20)
    else:
        bag = bow_model()
    test(classes = classes_train, directory = directory_test, bw = bag, no_clusters = 20)

main()

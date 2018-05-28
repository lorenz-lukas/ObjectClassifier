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

sift_object = cv2.xfeatures2d.SIFT_create()

def bow(directory,classes):
    global sift_object
    imlist = []
    name_dic = {}
    train_labels = np.array([])
    descriptor_list = []
    no_clusters = 20
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
            cv2.imshow("im", images[i])
            cv2.waitKey(20)
            train_labels = np.append(train_labels, label_count)
            kp, des = sift_object.detectAndCompute(images[i], None)
            descriptor_list.append(des)
        counter+=1

    cv2.destroyAllWindows()
    # perform clustering
    print "\nClustering Features\n"
    bov_descriptor_stack = formatND(descriptor_list)
    print "Using Kmeans to Performe Clustering\n"
    k = KMeans(n_clusters = no_clusters)
    kmeans_ret = k.fit_predict(bov_descriptor_stack)
    print "Preparing the Vocabulary\n"
    vocab = developVocabulary(n_clusters = no_clusters, n_images = len(imlist), descriptor_list = descriptor_list, kmeans_model = kmeans_ret)
    print "Data Normalization\n"
    # show vocabulary trained
    plotHist(vocabulary = vocab, n_clusters = no_clusters)
    # Data normalization
    if std is None:
        scale = StandardScaler().fit(vocab)
        vocab = scale.transform(vocab)
    else:
        print "STD not none. External STD supplied"
        vocab = std.transform(vocab)
    model = train(vocabulary = vocab, labels = train_labels)
    saveDictionary(m = model,v = vocab)
    return model, vocab

def formatND(l):
    """
    restructures list into vstack array of shape
    M samples x N features for sklearn
    """
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
    print "Training SVM"
    print SVC()
    #print "Train labels", labels
    model = SVC().fit(vocabulary, labels)
    print "Training completed"
    return model

def saveDictionary(m,v):
    data = {
            'model': m,
            'vocabulary': v
    }
    print data
    with open('dictionary.txt', 'w') as outfile:
        json.dump(data, outfile)
############################### TEST METHODS ##################################
def loadDictionary():
    with open('dictionary.txt') as json_file:
        data = json.load(json_file)
    print data
    return data

def test(directory, classes, bw):
    for c in xrange(len(classes)):
        (db, Folders, img_list) = os.walk(directory+classes[c]+'/').next()
        for i in xrange(len(img_list)):
            img = cv2.imread(directory+classes[c]+'/'+img_list[i])
            imlist.append(img)
    predictions = []
    for word, imagelist in imlist.iteritems():
        print "processing " ,word
        for im in imlist:
            # print imlist[0].shape, imlist[1].shape
            print im.shape
            cl = recognize(im)
            print cl
            predictions.append({
                'image':im,
                'class':cl,
                'object_name':name_dict[str(int(cl[0]))]
                })

    print predictions
    for each in predictions:
        # cv2.imshow(each['object_name'], each['image'])
        # cv2.waitKey()
        # cv2.destroyWindow(each['object_name'])
        #
        plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
        plt.title(each['object_name'])
        plt.show()

def recognize(test_img):
    kp, des = self.im_helper.features(test_img)
    # print kp
    print des.shape
    # generate vocab for test image
    vocab = np.array( [[ 0 for i in range(self.no_clusters)]])
    # locate nearest clusters for each of
    # the visual word (feature) present in the image
    # test_ret =<> return of kmeans nearest clusters for N features
    test_ret = self.bov_helper.kmeans_obj.predict(des)
    # print test_ret
    # print vocab
    for each in test_ret:
        vocab[0][each] += 1
    print vocab
    # Scale the features
    vocab = self.bov_helper.scale.transform(vocab)
    # predict the class of the image
    lb = self.bov_helper.clf.predict(vocab)
    # print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
    return lb

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print "################################################################"
    print "                 Bag of Visual Words                            "
    print "           Opencv 3.4.1 -- python 2.7                           "
    print "################################################################"
    print "\n\n"
    directory_train = 'CalTech101/Train/'
    directory_test = 'CalTech101/Test/'  # Same number of train classes
    model = os.listdir('/')
    print model
    classes_train = os.listdir(directory_train)
    bag = bow(directory = directory_train , classes = classes_train)
    #test(classes = classes_train, directory = directory_test, bw = bag)

main()

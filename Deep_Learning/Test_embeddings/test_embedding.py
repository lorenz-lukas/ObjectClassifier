import pickle
import cv2
import numpy as np

tr_images, tr_labels, tr_embeddings = pickle.load( open( 'train.pkl', "rb" ) )
te_images, te_labels, te_embeddings = pickle.load( open( 'test.pkl', "rb" ) )
ev_images, ev_labels, ev_embeddings = pickle.load( open( 'eval.pkl', "rb" ) )

a=1

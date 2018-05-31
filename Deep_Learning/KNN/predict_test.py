import pickle
import cv2
import numpy as np
import knn
import sys
import os

embedding_folder = sys.argv[1]
output_filename =  sys.argv[2]
input_pickle = sys.argv[3]
txt_out_filename = sys.argv[4]

def predict(n_dis, n_indx, labels, true_label):


    # delete_indexes = []
    # for idx, i in enumerate(n_dis):
    #     if n_dis[idx] > MARGIN:
    #         delete_indexes.append(idx)

    # for i in delete_indexes:
    #     n_dis = np.delete(n_dis, delete_indexes)
    #     n_indx = np.delete(n_indx, delete_indexes)

    # if n_dis.shape[0] == 0:
    #     return 0


    pred_labels = np.zeros(n_indx.shape[0], dtype=np.int32)
    for idx, i in enumerate(n_indx):
        pred_labels[idx] = labels[i]

    counts = np.bincount(pred_labels)
    max_freq = np.argmax(counts)

    max_freqs = np.nonzero(counts == counts[max_freq])
    max_freqs = max_freqs[0]

    if max_freqs.shape[0] > 1:
        candidate_labels = []
        candidate_avg = []
        n_appearences = float(counts[max_freq])  
        for i in range(max_freqs.shape[0]):
            candidate_label = max_freqs[i]
            candidate_labels.append(candidate_label)
            avg = 0.                    
            for j in range(pred_labels.shape[0]):
                if pred_labels[j] == candidate_label:
                    avg += (n_dis[j])/n_appearences
            candidate_avg.append(avg)

        lowest_avg_idx = candidate_avg.index(min(candidate_avg))
        a = 1
        return candidate_labels[lowest_avg_idx] == true_label
        

    else:
        a = 1
        return max_freqs[0] == true_label

_, best_knn, _ = pickle.load(open(input_pickle, 'rb')) 

MARGIN = 1
N_CLASSES = 102
images, labels, embeddings = pickle.load( open( os.path.join(embedding_folder,'train.pkl'), "rb" ) )
t_images, t_labels, t_embeddings = pickle.load( open( os.path.join(embedding_folder,'test.pkl'), "rb" ) )

transpose_embd = np.transpose(embeddings)
transpose_t_embd = np.transpose(t_embeddings)

best_acc = 0.0

by_class_acc = [np.array([])]*N_CLASSES

distances, indices = knn.knn(transpose_t_embd, transpose_embd, best_knn)


indices = indices - 1

n_test = t_labels.shape[0]
count = 0
    

for i in range(n_test):
    new_p = predict(np.transpose(distances[:,i]), np.transpose(indices[:,i]),labels, t_labels[i])
    count += new_p
    by_class_acc[ int(t_labels[i]) ] = np.append( by_class_acc[ int(t_labels[i]) ], [new_p] )

acc = float(count)/n_test

best_acc = acc
best_by_class_acc = by_class_acc

pickle.dump([best_acc, best_by_class_acc], open(output_filename, 'wb')) 

out_file_obj  = open(txt_out_filename, 'w')
out_file_obj.write('Accuracy: %f'%(best_acc))

out_file_obj.close()
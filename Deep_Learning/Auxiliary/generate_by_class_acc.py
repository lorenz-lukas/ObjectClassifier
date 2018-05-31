import os
import sys
import random

import pickle

import numpy as np



pickle_filename = sys.argv[1]
out_txt = sys.argv[2]

_, best_by_class_acc = pickle.load( open( pickle_filename, "rb" ) )

temp_vec = np.zeros([len(best_by_class_acc)])

for index, bc_evals in enumerate(best_by_class_acc):
    temp_vec[index] = float(np.sum(bc_evals) )/bc_evals.shape[0]

acc = np.mean(temp_vec)
std = np.std(temp_vec)


out_file_obj  = open(out_txt, 'w')
out_file_obj.write('By class accuracy: %f\n'%(acc))
out_file_obj.write('Stdev: %f'%(std))

out_file_obj.close()
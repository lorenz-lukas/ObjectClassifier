from sklearn.model_selection import KFold, cross_val_score
import os
import re
import pickle

train_filenames = []

basefolder = '../Dataset/Train'

all_dirs = [i for i in os.listdir(basefolder) if os.path.isdir(os.path.join(basefolder,i))]
all_dirs.sort()

jpg_pattern = re.compile('\w*\.jpg')

n_splits = 5

train_split = [[]]*n_splits
train_split_classes = [[]]*n_splits
eval_split = [[]]*n_splits
eval_split_classes = [[]]*n_splits
for s_class,s_dir in enumerate(all_dirs):
    full_dir = os.path.join(basefolder,s_dir)
    all_files = [i for i in os.listdir(full_dir) if jpg_pattern.search(i) is not None]
    to_be_split = [os.path.join(s_dir,i) for i in all_files]
    k_fold = KFold(n_splits=n_splits, shuffle=True)
    idx = 0
    for train_indices, eval_indices in k_fold.split(to_be_split):
        new_train_split = [to_be_split[i] for i in train_indices]
        new_train_split_classes = [s_class]*len(new_train_split)
        new_eval_split = [to_be_split[i] for i in eval_indices]
        new_eval_split_classes = [s_class]*len(new_eval_split)
        train_split[idx] = train_split[idx] + new_train_split
        train_split_classes[idx] = train_split_classes[idx] + new_train_split_classes
        eval_split[idx] = eval_split[idx] + new_eval_split
        eval_split_classes[idx] = eval_split_classes[idx] + new_eval_split_classes        
        idx = idx + 1



for single_split in range(n_splits):
    pickle.dump([train_split[single_split], train_split_classes[single_split], eval_split[single_split], eval_split_classes[single_split]], open('split_%02d.pkl'%(single_split), 'wb')) 
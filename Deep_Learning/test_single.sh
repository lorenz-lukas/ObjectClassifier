#!/bin/bash +ex

#Input: folder in Recently_evaluated (with best and current) + .py in Ongoing_Models
#Change name in 'folder' below

modelBase="Ongoing_Models"
folder="Ongoing_Models/2smallresnet_he_with_expdecay_nadam"


evaluatedPath="./Recently_evaluated/"
tfrecordsPath="./TFRecords_files/Current_sets/"
out_txt="acc_log.txt"

tfrecords_filenames=( "32_by_32_eval_set.tfrecords" "cifar10_eval.tfrecords")



model_folder_no_path=$(basename -- "$folder")
if [ ! -d "$evaluatedPath$model_folder_no_path" ]; then
    mkdir "$evaluatedPath$model_folder_no_path"
fi

temp=($folder/*_train.py)
train_file=${temp[0]}

temp=($folder/*_model.py)
model_file=${temp[0]}

model_file_no_path=$(basename -- "$model_file") #Removing dirname from string
model="${model_file_no_path%.*}" # Removing extension


for j in "${tfrecords_filenames[@]}"
    do
        python Eval_code/eval_multi_gpu_no_moving_average.py $evaluatedPath$model_folder_no_path/current $tfrecordsPath$j $evaluatedPath$model_folder_no_path/$out_txt $modelBase.$model_folder_no_path.$model
    done



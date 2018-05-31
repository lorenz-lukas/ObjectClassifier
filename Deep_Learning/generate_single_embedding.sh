#!/bin/bash +ex

split_size=05
j=00

model_folder_no_path="10_multi_gpu_expdecay_nadam_with_smallerresnet_he_triplet_model_new_cost"
embeddingPath="./Embeddings/"
evaluatedPath="./Recently_evaluated/"
tfrecordsPath="./TFRecords_files/"
model="smallerresnet_he_triplet_model"

if [ -d "$embeddingPath${model_folder_no_path}_split_${split_size}_$j" ]; then
    rm -rf "$embeddingPath${model_folder_no_path}_split_${split_size}_$j"
fi
mkdir "$embeddingPath${model_folder_no_path}_split_${split_size}_$j"  

input_dir="${evaluatedPath}${model_folder_no_path}_split_${split_size}_$j/current"
out_filename=("$embeddingPath${model_folder_no_path}_split_${split_size}_$j/test.pkl" "$embeddingPath${model_folder_no_path}_split_${split_size}_$j/train.pkl" "$embeddingPath${model_folder_no_path}_split_${split_size}_$j/eval.pkl")
tfrecords_filename=("${tfrecordsPath}Splits_${split_size}/test.tfrecords" "${tfrecordsPath}Splits_${split_size}/split_${j}_train.tfrecords" "${tfrecordsPath}Splits_${split_size}/split_${j}_eval.tfrecords")

for index in "${!out_filename[@]}"; do
    python ./Generate_embeddings/generate_embeddings.py $input_dir ${tfrecords_filename[index]} ${out_filename[index]} $model
done

knn_max=$split_size

python KNN/predict_eval.py $embeddingPath${model_folder_no_path}_split_${split_size}_$j $evaluatedPath${model_folder_no_path}_split_${split_size}_$j/eval_results.pkl $knn_max $evaluatedPath${model_folder_no_path}_split_${split_size}_$j/eval_results.txt
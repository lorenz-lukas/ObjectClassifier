#!/bin/bash +ex

modelBase="Ongoing_Models"
modelFolders=($modelBase/*_with_*)


evaluatedPath="./Recently_evaluated/"
tfrecordsPath="./TFRecords_files/Current_sets/"
out_txt="acc_moving_average_log.txt"
out_nmav_txt="acc_no_moving_average_log.txt"

tfrecords_filenames=( "32_by_32_eval_set.tfrecords" "cifar10_eval.tfrecords")

processed=()

while ! [ ${#modelFolders[@]} -eq 0 ];
    do
        for folder in "${modelFolders[@]}"
            do
                if ! [[ " ${processed[*]} " == *" $folder "* ]]; then
                    processed+=( "$folder" )
                    break                    
                fi
            done


        model_folder_no_path=$(basename -- "$folder")
        if [ -d "$evaluatedPath$model_folder_no_path" ]; then
            rm -rf "$evaluatedPath$model_folder_no_path"
        fi
        mkdir "$evaluatedPath$model_folder_no_path"

        temp=($folder/*_train.py)
        train_file=${temp[0]}

        temp=($folder/*_model.py)
        model_file=${temp[0]}

        model_file_no_path=$(basename -- "$model_file") #Removing dirname from string
        model="${model_file_no_path%.*}" # Removing extension

        python $train_file $model
        mv ./Current_training/* $evaluatedPath$model_folder_no_path

        for j in "${tfrecords_filenames[@]}"
            do
                python Eval_code/eval_multi_gpu.py $evaluatedPath$model_folder_no_path/best $tfrecordsPath$j $evaluatedPath$model_folder_no_path/$out_txt $modelBase.$model_folder_no_path.$model
                python Eval_code/eval_multi_gpu.py $evaluatedPath$model_folder_no_path/current $tfrecordsPath$j $evaluatedPath$model_folder_no_path/$out_txt $modelBase.$model_folder_no_path.$model
                python Eval_code/eval_multi_gpu_no_moving_average.py $evaluatedPath$model_folder_no_path/best $tfrecordsPath$j $evaluatedPath$model_folder_no_path/$out_nmav_txt $modelBase.$model_folder_no_path.$model
                python Eval_code/eval_multi_gpu_no_moving_average.py $evaluatedPath$model_folder_no_path/current $tfrecordsPath$j $evaluatedPath$model_folder_no_path/$out_nmav_txt $modelBase.$model_folder_no_path.$model
            done

        #Reload models (in case user changed any)
        modelFolders=($modelBase/*_with_*)
        for target in "${processed[@]}"; do
            for index in "${!modelFolders[@]}"; do
                if [[ ${modelFolders[index]} = $target ]]; then
                    unset 'modelFolders[index]'
                fi
            done
        done         

    done

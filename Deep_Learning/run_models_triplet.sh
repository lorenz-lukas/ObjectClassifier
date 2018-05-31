#!/bin/bash +ex

modelBase="Ongoing_Models"
modelFolders=($modelBase/*_with_*)

embeddingPath="./Embeddings/"
evaluatedPath="./Recently_evaluated/"
tfrecordsPath="./TFRecords_files/"
out_txt="acc_moving_average_log.txt"
out_nmav_txt="acc_no_moving_average_log.txt"

split_sizes=(03 05 10 15 20 30)
tfrecords_indexes=(00)

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


        for split_size in "${split_sizes[@]}"; do
            for j in "${tfrecords_indexes[@]}"; do
                

                model_folder_no_path=$(basename -- "$folder")
                if [ -d "$evaluatedPath${model_folder_no_path}_split_${split_size}_$j" ]; then
                    rm -rf "$evaluatedPath${model_folder_no_path}_split_${split_size}_$j"
                fi
                mkdir "$evaluatedPath${model_folder_no_path}_split_${split_size}_$j"

                temp=($folder/*_valid_train.py)
                train_file=${temp[0]}

                temp=($folder/*_full_train.py)
                full_train_file=${temp[0]}

                temp=($folder/*_model.py)
                model_file=${temp[0]}

                model_file_no_path=$(basename -- "$model_file") #Removing dirname from string
                model="${model_file_no_path%.*}" # Removing extension
                

                num_steps_file=$evaluatedPath${model_folder_no_path}_split_${split_size}_$j/num_steps.pkl



                python $train_file $model $split_size $j $num_steps_file

                
                mv ./Current_training/* $evaluatedPath${model_folder_no_path}_split_${split_size}_$j
                cp $folder $evaluatedPath${model_folder_no_path}_split_${split_size}_$j


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



                if [ -d "$evaluatedPath${model_folder_no_path}_split_${split_size}_full" ]; then
                    rm -rf "$evaluatedPath${model_folder_no_path}_split_${split_size}_full"
                fi
                mkdir "$evaluatedPath${model_folder_no_path}_split_${split_size}_full"

                python $full_train_file $model $split_size $num_steps_file               


                mv ./Current_training/* $evaluatedPath${model_folder_no_path}_split_${split_size}_full


                if [ -d "$embeddingPath${model_folder_no_path}_split_${split_size}_full" ]; then
                    rm -rf "$embeddingPath${model_folder_no_path}_split_${split_size}_full"
                fi
                mkdir "$embeddingPath${model_folder_no_path}_split_${split_size}_full"  

                input_dir="$evaluatedPath${model_folder_no_path}_split_${split_size}_full/current"
                out_filename=("$embeddingPath${model_folder_no_path}_split_${split_size}_full/test.pkl" "$embeddingPath${model_folder_no_path}_split_${split_size}_full/train.pkl")
                tfrecords_filename=("${tfrecordsPath}Splits_${split_size}/test.tfrecords" "${tfrecordsPath}Splits_${split_size}/full_train.tfrecords")

                for index in "${!out_filename[@]}"; do
                    python ./Generate_embeddings/generate_embeddings.py $input_dir ${tfrecords_filename[index]} ${out_filename[index]} $model
                done


                python KNN/predict_test.py $embeddingPath${model_folder_no_path}_split_${split_size}_full $evaluatedPath${model_folder_no_path}_split_${split_size}_full/test_results.pkl $evaluatedPath${model_folder_no_path}_split_${split_size}_$j/eval_results.pkl $evaluatedPath${model_folder_no_path}_split_${split_size}_full/test_results.txt
                python Auxiliary/generate_by_class_acc.py $evaluatedPath${model_folder_no_path}_split_${split_size}_full/test_results.pkl $evaluatedPath${model_folder_no_path}_split_${split_size}_full/by_class_results.txt

            done
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

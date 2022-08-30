#!/bin/bash

# file list
# seg-first-dependent-main-last
# seg-first-independent-main-last
# split-first-dependent-main-last
# split-first-independent-main-last

run_condition="split-first-dependent-main-last"
# touch $run_condition.log
# jupyter nbconvert --to script *.ipynb


models=("LSTM" "Conv1D_LSTM" "Conv1D_LSTM_Attention" "CNN2D") 
stims=(0 1)
segment_numbers=(1 3 5 60)

for model in ${models[@]} ; do
    for segment_number in ${segment_numbers[@]} ; do
        for stim in ${stims[@]} ; do
            # echo "------------------------------------------------------------" >> $run_condition.log
            # echo $model , $segment_number, $stim >> $run_condition.log
            echo python3 -u $run_condition.py --model_name $model --stim $stim --segment_number $segment_number --len_reduction None --isdebug False
            # echo ""
        done
    done
done


models=("Conv1D_LSTM_SelfAttention") 
stims=(0 1)
len_reductions=("mean")
segment_numbers=(1 3 5 60)

for model in ${models[@]} ; do
    for segment_number in ${segment_numbers[@]} ; do
        for stim in ${stims[@]} ; do
            for len_reduction in ${len_reductions[@]} ; do
                # echo "------------------------------------------------------------" >> $run_condition.log
                # echo $model , $segment_number, $stim >> $run_condition.log
                echo python3 -u $run_condition.py --model_name $model --stim $stim --segment_number $segment_number --len_reduction $len_reduction --isdebug False
                # echo ""
            done
        done
    done
done
#/bin/bash
# for seg in 60 20 12 1
for seg in 60 20
do
    for preprocessing in "DE" "DASM" "RASM" "DCAU" "PCC_TIME" "PCC_FREQ" "PLV" "PLI"
    do
        for sti in "arousal" "valence"
        do
            for exp in "trial" "segment"
            do
                for sub in "dependent" "independent"
                do
                    name="./output/"$sub-$exp-$sti-$preprocessing-$seg".log" 
                    cmd="python3 main-ml.py \
                            --subject_setup $sub \
                            --experimental_setup $exp \
                            --stimuli_class $sti \
                            --preprocessing $preprocessing \
                            --segment_lenght $seg\
                            --output_log $name"
                    final=`grep final $name | wc -l`
                    if [ $final -gt 0 ]; then
                        echo "$name is done. final = $final"
                        continue
                    fi
                    # if test -f "$name"; then
                    #     echo "$name exists."
                    #     cmd="$cmd --continue 1"
                    # fi
                    echo $cmd
                    start=$(date +%s)
                    $cmd
                    end=$(date +%s)
                    echo "run time: $(($end-$start)) seconds"
                done
            done
        done
    done
done
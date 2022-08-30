#/bin/bash
preprocessing="PLI"
for seg in 1
do
    # for sub in "dependent" "independent"
    for sub in "independent"
    do
        for sti in "arousal" "valence"
        do
            # for exp in "trial" "segment"
            for exp in "trial"
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
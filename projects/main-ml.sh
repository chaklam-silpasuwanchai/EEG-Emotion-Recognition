#/bin/bash
preprocessing="PCC_TIME"
for seg in 60 20 12 1
do
    for sub in "dependent" "independent"
    do
        for sti in "arousal" "valence"
        do
            for exp in "trial" "segment"
            do
                name="./output/"$sub-$exp-$sti-$preprocessing-$seg".log" 
                if test -f "$name"; then
                    echo "$name exists."
                    continue 
                fi
                cmd="python3 main-ml.py \
                        --subject_setup $sub \
                        --experimental_setup $exp \
                        --stimuli_class $sti \
                        --preprocessing $preprocessing \
                        --segment_lenght $seg\
                        --output_log $name"
                echo $cmd
                start=$(date +%s)
                $cmd
                end=$(date +%s)
                echo "run time: $(($end-$start)) seconds"
            done
        done
    done
done
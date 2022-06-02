#/bin/bash
preprocessing="DE"
for seg in 60 20 12 1
do
    for sub in "dependent" "independent"
    do
        for sti in "arousal" "valence"
        do
            for exp in "trial" "segment"
            do
                cmd="python3 -u main-ml.py \
                        --subject_setup $sub \
                        --experimental_setup $exp \
                        --stimuli_class $sti \
                        --preprocessing $preprocessing \
                        --segment_lenght $seg\
                        --output_log ./output/$sub\_$exp\_$sti\_$preprocessing\_$seg.log"
                echo $cmd
                start=$(date +%s)
                $cmd
                end=$(date +%s)
                echo "run time: $(($end-$start)) seconds"
            done
        done
    done
done
#!/bin/bash
# rm ./output/*.log
python3 main-ml.py --subject_setup dependent --experimental_setup trial --stimuli_class valence --preprocessing DASM --segment_lenght 60 --output_log ./output/dependent_trial_valence_DASM_60.log
python3 main-ml.py --subject_setup dependent --experimental_setup trial --stimuli_class valence --preprocessing DASM --segment_lenght 20 --output_log ./output/dependent_trial_valence_DASM_20.log
python3 main-ml.py --subject_setup dependent --experimental_setup trial --stimuli_class valence --preprocessing DASM --segment_lenght 12 --output_log ./output/dependent_trial_valence_DASM_12.log
python3 main-ml.py --subject_setup dependent --experimental_setup trial --stimuli_class valence --preprocessing DASM --segment_lenght 1 --output_log ./output/dependent_trial_valence_DASM_1.log
python3 main-ml.py --subject_setup dependent --experimental_setup trial --stimuli_class arousal --preprocessing DASM --segment_lenght 60 --output_log ./output/dependent_trial_arousal_DASM_60.log
python3 main-ml.py --subject_setup dependent --experimental_setup trial --stimuli_class arousal --preprocessing DASM --segment_lenght 20 --output_log ./output/dependent_trial_arousal_DASM_20.log
python3 main-ml.py --subject_setup dependent --experimental_setup trial --stimuli_class arousal --preprocessing DASM --segment_lenght 12 --output_log ./output/dependent_trial_arousal_DASM_12.log
python3 main-ml.py --subject_setup dependent --experimental_setup trial --stimuli_class arousal --preprocessing DASM --segment_lenght 1 --output_log ./output/dependent_trial_arousal_DASM_1.log

python3 main-ml.py --subject_setup dependent --experimental_setup segment --stimuli_class valence --preprocessing DASM --segment_lenght 60 --output_log ./output/dependent_segment_valence_DASM_60.log
python3 main-ml.py --subject_setup dependent --experimental_setup segment --stimuli_class valence --preprocessing DASM --segment_lenght 20 --output_log ./output/dependent_segment_valence_DASM_20.log
python3 main-ml.py --subject_setup dependent --experimental_setup segment --stimuli_class valence --preprocessing DASM --segment_lenght 12 --output_log ./output/dependent_segment_valence_DASM_12.log
python3 main-ml.py --subject_setup dependent --experimental_setup segment --stimuli_class valence --preprocessing DASM --segment_lenght 1 --output_log ./output/dependent_segment_valence_DASM_1.log
python3 main-ml.py --subject_setup dependent --experimental_setup segment --stimuli_class arousal --preprocessing DASM --segment_lenght 60 --output_log ./output/dependent_segment_arousal_DASM_60.log
python3 main-ml.py --subject_setup dependent --experimental_setup segment --stimuli_class arousal --preprocessing DASM --segment_lenght 20 --output_log ./output/dependent_segment_arousal_DASM_20.log
python3 main-ml.py --subject_setup dependent --experimental_setup segment --stimuli_class arousal --preprocessing DASM --segment_lenght 12 --output_log ./output/dependent_segment_arousal_DASM_12.log
python3 main-ml.py --subject_setup dependent --experimental_setup segment --stimuli_class arousal --preprocessing DASM --segment_lenght 1 --output_log ./output/dependent_segment_arousal_DASM_1.log

python3 main-ml.py --subject_setup independent --experimental_setup trial --stimuli_class valence --preprocessing DASM --segment_lenght 60 --output_log ./output/independent_trial_valence_DASM_60.log
python3 main-ml.py --subject_setup independent --experimental_setup trial --stimuli_class valence --preprocessing DASM --segment_lenght 20 --output_log ./output/independent_trial_valence_DASM_20.log
python3 main-ml.py --subject_setup independent --experimental_setup trial --stimuli_class valence --preprocessing DASM --segment_lenght 12 --output_log ./output/independent_trial_valence_DASM_12.log
python3 main-ml.py --subject_setup independent --experimental_setup trial --stimuli_class valence --preprocessing DASM --segment_lenght 1 --output_log ./output/independent_trial_valence_DASM_1.log
python3 main-ml.py --subject_setup independent --experimental_setup trial --stimuli_class arousal --preprocessing DASM --segment_lenght 60 --output_log ./output/independent_trial_arousal_DASM_60.log
python3 main-ml.py --subject_setup independent --experimental_setup trial --stimuli_class arousal --preprocessing DASM --segment_lenght 20 --output_log ./output/independent_trial_arousal_DASM_20.log
python3 main-ml.py --subject_setup independent --experimental_setup trial --stimuli_class arousal --preprocessing DASM --segment_lenght 12 --output_log ./output/independent_trial_arousal_DASM_12.log
python3 main-ml.py --subject_setup independent --experimental_setup trial --stimuli_class arousal --preprocessing DASM --segment_lenght 1 --output_log ./output/independent_trial_arousal_DASM_1.log

python3 main-ml.py --subject_setup independent --experimental_setup segment --stimuli_class valence --preprocessing DASM --segment_lenght 60 --output_log ./output/independent_segment_valence_DASM_60.log
python3 main-ml.py --subject_setup independent --experimental_setup segment --stimuli_class valence --preprocessing DASM --segment_lenght 20 --output_log ./output/independent_segment_valence_DASM_20.log
python3 main-ml.py --subject_setup independent --experimental_setup segment --stimuli_class valence --preprocessing DASM --segment_lenght 12 --output_log ./output/independent_segment_valence_DASM_12.log
python3 main-ml.py --subject_setup independent --experimental_setup segment --stimuli_class valence --preprocessing DASM --segment_lenght 1 --output_log ./output/independent_segment_valence_DASM_1.log
python3 main-ml.py --subject_setup independent --experimental_setup segment --stimuli_class arousal --preprocessing DASM --segment_lenght 60 --output_log ./output/independent_segment_arousal_DASM_60.log
python3 main-ml.py --subject_setup independent --experimental_setup segment --stimuli_class arousal --preprocessing DASM --segment_lenght 20 --output_log ./output/independent_segment_arousal_DASM_20.log
python3 main-ml.py --subject_setup independent --experimental_setup segment --stimuli_class arousal --preprocessing DASM --segment_lenght 12 --output_log ./output/independent_segment_arousal_DASM_12.log
python3 main-ml.py --subject_setup independent --experimental_setup segment --stimuli_class arousal --preprocessing DASM --segment_lenght 1 --output_log ./output/independent_segment_arousal_DASM_1.log
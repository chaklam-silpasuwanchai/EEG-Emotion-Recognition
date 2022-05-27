from curses import meta
from time import time
from components.dataset_jo import Dataset_subjectDependent as MyDataset
from components.ml import train_model_segment_first, train_model_split_first
from components.preprocessing import standardize, DE, ASYM
import os
import logging
import argparse
import numpy as np

def get_argument() -> argparse.Namespace:
    # python filename -h description
    parser = argparse.ArgumentParser(description="Running EEG Emotion recognition using various preprocessings")
    parser.add_argument('--segment_lenght','-sl', metavar='SL',type=int,default=60,                    help="The window sizes of segmentation in second.")
    parser.add_argument('--output_log','-o',      metavar='OL',type=str,default='./output/main-ml.log',help="Path including filename of the output.")
    parser.add_argument('--stimuli_class','-sc',  metavar='SC',type=str,default='valence',             help="Class of stimuli of classification. It either be 'valence' or 'arousal'.")
    parser.add_argument('--preprocessing','-p',   metavar='P' ,type=str,default='DE',                  help=f"The type of preprocessing. Possible option are {preprocessing_option}.")
    parser.add_argument('--subject_setup','-ss',  metavar='SJ',type=str,default='dependent',           help="The configuration when building model. 'dependent' will train model for each subject. 'independent' will train one model for all subjects.")
    parser.add_argument('--experimental_setup','-es', metavar='EX',type=str,default='trial',           help="The configuration when spliting data. 'trial' will split data such that no single trial exist in both train and test. 'segment' assume each segment is independent, thus segments from one trial can exist in both train and test.")
    # TODO: write parameters 
    return parser.parse_args()


if __name__ == '__main__':
    preprocessing_option = ['DE','DASM','RASM','DCAU']
    args = get_argument()
    # Start logging
    logging.basicConfig(filename=f'{args.output_log}',
                    format='%(asctime)s|%(levelname)s|%(message)s', 
                    datefmt='%d-%m-%Y %H:%M:%S',
                    level=logging.INFO)
    logging.info(f'''
    Running Command: 
        python3 main-ml.py --subject_setup {args.subject_setup} --experimental_setup {args.experimental_setup} --stimuli_class {args.stimuli_class} --preprocessing {args.preprocessing} --segment_lenght {args.segment_lenght} --output_log {args.output_log}
    Starting training:
        segment_lenght:  {args.segment_lenght}
        output_log:      {args.output_log}
        stimuli_class:   {args.stimuli_class}
        preprocessing:   {args.preprocessing}
        subject_setup:   {args.subject_setup}
        experimental_setup: {args.experimental_setup}
    ''')
    # Check input parameter
    assert args.preprocessing in preprocessing_option,          f"The preprocessing '{args.preprocessing}' is not supported."
    assert args.stimuli_class in ['valence','arousal'],         f"The stimuli_class '{args.stimuli_class}' is not supported."
    assert args.subject_setup in ['dependent', 'independent'],  f"The subject_setip '{args.subject_setup}' is not supported."
    assert args.experimental_setup in ['trial', 'segment'],     f"The experimental_setup '{args.experimental_setup}' is not supported."

    # Load dataset from path ./data. Inside the path must be s01, s02, s03, ...
    # Lazyload mean the class will not load data if not use. This will save some RAM. 
    # But it will eventually load all the data because I did not write garbage collector
    dataset = MyDataset(dataset_path='data', lazyload=True)
    dataset.set_segment(7680//(128*args.segment_lenght))

    stimuli_class = 0
    if(args.stimuli_class == 'valence'):
        stimuli_class = MyDataset.STIMULI_VALENCE
    elif(args.stimuli_class == 'arousal'):
        stimuli_class = MyDataset.STIMULI_AROUSAL

    # init output folder
    output_gridsearch_path = f"./output/gridSearch-{args.subject_setup}-{args.experimental_setup}-{args.stimuli_class}-{args.preprocessing}-{args.segment_lenght}"
    if(os.path.exists(output_gridsearch_path) == False):
        os.mkdir(output_gridsearch_path)

    # assign preprocessing
    preprocessing = None
    if(args.preprocessing == 'DE'):
        preprocessing = DE
    elif(args.preprocessing in ['DASM','RASM','DCAU']):
        preprocessing = ASYM


    experimental_setup = None
    if(args.experimental_setup == 'trial'):
        experimental_setup = train_model_segment_first
    elif(args.experimental_setup == 'segment'):
        experimental_setup = train_model_split_first

    cv_scores_final, cv_scores_std_final = [], []
    if(args.subject_setup == 'dependent'):
        for filename in dataset.get_file_list():
            start = time()
            data, labels, groups = dataset.get_data(filename, stimuli=stimuli_class, return_type='numpy')

            X = preprocessing(data, variant=args.preprocessing)
            X = standardize(X)
            # if experimental_setup is split_first, groups will be ignored
            cv_scores = experimental_setup(X, labels.reshape(-1), groups, cv_result_prefix=f"{output_gridsearch_path}/{filename}")

            logging.info(f"{filename}|10-CV={format(  round(cv_scores.mean(),5), '.5f')}|STD={format(  round(cv_scores.std(),5), '.5f')}|Time spend={time() - start}")
            cv_scores_final.append(cv_scores.mean())
            cv_scores_std_final.append(cv_scores.std())
    elif(args.subject_setup == 'independent'):
        # Get all subj in one big datas, labels, groups
        all_datas, all_labels, all_groups = [],[],[]
        for filename in dataset.get_file_list():
            data, labels, groups = dataset.get_data(filename, stimuli=0, return_type='numpy')
            # subj 1 will have groups start from 100 101 102 ... 139
            # subj 2 will have groups start from 200 201 202 ... 239
            # ...
            # subj 32 will have groups start from 3200 3201 3202 ... 3239
            groups = int(filename[1:])*100 +  groups
            # print(filename, int(filename[1:])*100 +  groups)
            all_datas.append(data)
            all_labels.append(labels)
            all_groups.append(groups.reshape(-1,1))
        all_datas = np.vstack(all_datas)
        all_labels = np.vstack(all_labels).reshape(-1)
        all_groups = np.vstack(all_groups).reshape(-1)

        start = time()
        X = preprocessing(data, variant=args.preprocessing)
        X = standardize(X)
        # if experimental_setup is split_first, groups will be ignored
        cv_scores = experimental_setup(X, all_labels.reshape(-1), all_groups, cv_result_prefix=f"{output_gridsearch_path}/{filename}")

        logging.info(f"ALL|10-CV={format(  round(cv_scores.mean(),5), '.5f')}|STD={format(  round(cv_scores.std(),5), '.5f')}|Time spend={time() - start}")
        cv_scores_final.append(cv_scores.mean())
        cv_scores_std_final.append(cv_scores.std())


    cv_scores_final = np.array(cv_scores_final)
    cv_scores_std_final = np.array(cv_scores_std_final)
    logging.info(f"final|10-CV={format(  round(cv_scores_final.mean(),5), '.5f')}|STD={format(  round(cv_scores_std_final.std(),5), '.5f')}")
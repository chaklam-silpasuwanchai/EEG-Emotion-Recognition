# EEG-Emotion-Recognition

This tutorial covers most traditional machine learning algorithms and deep learning algorithms for predicting EEG signals.  

My intention is that this can be used by my Master and Ph.D. students as the getting started kit for their EEG research.   These algorithms may not necessary work well particularly for EEG emotion recognition, but I have include all of them for the sake of completeness.

Note: Before using the tutorials, please create a folder "data" and download preprocessed DEAP dataset and put s01.dat,...,s32.dat inside this "data" folder.  The data folder will be in the same directory as the tutorial.

This tutorial explains basic EEG analysis as well as common deep learning models, by using emotion recognition from the benchmark DEAP dataset as the case study.

Python libraries:
1. Python MNE
2. PyTorch
3. NumPy
4. Sckit-Learn

Tutorials:
1. Understanding the DEAP dataset (Done)
2. EEG feature engineering + machine learning
   - spectrum + SVM
   - asymmetry + SVM
   - connectivity + SVM
   - common spatial pattern + SVM
3. Deep learning models
   - LSTM (71% accuracy)
   - CNN1D + LSTM (89% accuracy)
   - CNN1D + LSTM + General Attention (92% accuracy)
   - CNN1D + LSTM + Self Attention (85% accuracy)
   - CNN1D + LSTM + MultiHead Attention (87% accuracy)
   - Spectogram + CNN2D
   - EEGNet
   

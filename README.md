# EEG-Emotion-Recognition

Note: Before using the tutorials, please create a folder "data" and download preprocessed DEAP dataset and put s01.dat,...,s32.dat inside this "data" folder.  The data folder will be in the same directory as the tutorial.

This tutorial explains basic EEG analysis as well as common deep learning models, by using emotion recognition from the benchmark DEAP dataset as the case study.

Python libraries:
1. Python MNE
2. PyTorch
3. NumPy
4. Sckit-Learn

Tutorials:
1. Understanding the DEAP dataset (Done)
2. EEG feature engineering
   - spectrum
   - asymmetry
   - connectivity
   - common spatial pattern
4. Machine learning models
   - SVM
5. Deep learning models
   - LSTM (51.56% accuracy)
   - CNN1D + LSTM (57.81% accuracy)
   - CNN1D + LSTM + General Attention (56.25% accuracy)
   - CNN1D + LSTM + Self Attention (46.88% accuracy)
   - Spectogram + CNN2D
   - EEGNet
   

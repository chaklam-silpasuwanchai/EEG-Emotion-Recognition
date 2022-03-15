# EEG-Emotion-Recognition

This tutorial covers most traditional machine learning algorithms and deep learning algorithms for predicting EEG signals.  

These tutorials are contributed by my Ph.D. student Mr. Akraradet, and Master students Ms. Pranissa, Ms. Chanapa, and Mr. Pongkorn.

My intention is that this can be used by my Master and Ph.D. students as the getting started kit for their EEG research.   These algorithms may not necessary work well particularly for EEG emotion recognition, but I have include all of them for the sake of completeness.

Note: Before using the tutorials, please create a folder "data" and download preprocessed DEAP dataset and put s01.dat,...,s32.dat inside this "data" folder.  The data folder will be in the same directory as the tutorial.

This tutorial explains basic EEG analysis as well as common deep learning models, by using emotion recognition from the benchmark DEAP dataset as the case study.

Python libraries:
1. Python MNE
2. PyTorch
3. NumPy
4. Sckit-Learn
5. SciPy

Tutorials:
1. Understanding the DEAP dataset (Done)
2. EEG feature engineering + machine learning
   - spectrum + SVM (61%)
   - asymmetry + SVM - Akraradet
   - connectivity + SVM - Akraradet
   - common spatial pattern + SVM - Akraradet
   - phase coherence
3. Deep learning models
   - LSTM (71% accuracy)
   - CNN1D + LSTM (89% accuracy)
   - CNN1D + LSTM + General Attention (91% accuracy)
   - CNN1D + LSTM + Self Attention (86% accuracy)
   - CNN1D + LSTM + MultiHead Attention (50 epochs - 93% accuracy)
   - Spectogram + CNN2D (50 epochs - 85%)
4. Advanced
   - CNN1D + LSTM + Hierarchical Attention (50 epochs - ) Fabby
   - CNN1D + LSTM + Multiplicative Attention (50 epochs - ) Babby
   - CNN1D + LSTM + Additive Attention (50 epochs - ) Babby
   - ChannelNet (50 epochs - ) Beau
   - EEGNet (50 epochs - ) Chaky
   - Spatial–Temporal Self-Attention CNN (50 epochs - ) New + Babby + Beau
   

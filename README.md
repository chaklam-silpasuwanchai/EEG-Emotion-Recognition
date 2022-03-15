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
1. (01.) Understanding the DEAP dataset (Done)
2. Common EEG feature engineering + machine learning
   - (02.) spectrum + SVM (61%)
   - (03.) asymmetry + SVM - Akraradet
   - (04.) connectivity + SVM - Akraradet
   - (05.) common spatial pattern + SVM - Akraradet
   - (06.) phase coherence + SVM - Akraradet
3. Baseline deep learning models
   - (07.) LSTM (50 epochs - 71% accuracy)
   - (08.) CNN1D + LSTM (50 epochs - 92% accuracy)
   - (09.) Spectogram + CNN2D 
     - CNN 2 layers - (50 epochs - 87% accuracy)
     - CNN 4 layers - (50 epochs - 96% accuracy)
4. Advanced
   - (10.) CNN1D + LSTM + General Attention 
     - Dot Product Attention (50 epochs - 94% accuracy)
     - Multiplicative Attention (50 epochs - ) Fabby
     - Additive Attention (50 epochs - ) Babby
     - Hierarchical Attention (50 epochs - ) Fabby
   - (11.) CNN1D + LSTM + Self Attention
     - Using mean to combine (50 epochs - 90% accuracy)
     - Using sum to combine (50 epochs - 91% accuracy)
     - Using last state to combine (50 epochs - 90% accuracy)
   - (12.) CNN1D + LSTM + MultiHead Attention 
     - Using mean to combine (50 epochs - 92% accuracy)
     - Using sum to combine (50 epochs - 92% accuracy)
     - Using last state to combine (50 epochs - 94% accuracy)
   - (13.) ChannelNet (50 epochs - ) Beau
   - (14.) EEGNet (50 epochs - ) Chaky
   - (15.) Spatial–Temporal Self-Attention CNN (50 epochs - ) New + Babby + Beau
   

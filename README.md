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
3. Baseline deep learning models (50 epochs)
   - (07.) LSTM (71% acc)
   - (08.) CNN1D + LSTM (92% acc)
   - (09.) Spectogram + CNN2D 
     - CNN 2 layers - (87% acc)
     - CNN 4 layers - (96% acc)
4. Advanced (50 epochs)
   - (10.) CNN1D + LSTM + General Attention 
     - Dot Product Attention (94% acc)
     - Multiplicative Attention (  ) Fabby
     - Additive Attention (  ) Babby
     - Hierarchical Attention (  ) Fabby
   - (11.) CNN1D + LSTM + Self Attention
     - Using mean to combine (90% accuracy)
     - Using sum to combine (91% accuracy)
     - Using last state to combine (90% accuracy)
   - (12.) CNN1D + LSTM + MultiHead Attention 
     - Using mean to combine (92% accuracy)
     - Using sum to combine (92% accuracy)
     - Using last state to combine (94% accuracy)
   - (13.) ChannelNet ( ) Beau
   - (14.) EEGNet ( ) Chaky
   - (15.) Spatial–Temporal Self-Attention CNN ( ) New + Babby + Beau
   

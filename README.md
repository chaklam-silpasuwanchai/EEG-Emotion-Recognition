# EEG-Emotion-Recognition

*Authors: myself and my Ph.D. student Mr. Akraradet, and Master students Ms. Pranissa, Ms. Chanapa, and Mr. Pongkorn.*

This tutorial covers basic EEG analyis, traditional machine learning algorithms and deep learning algorithms for predicting EEG signals, by using emotion recognition from the benchmark DEAP dataset as the case study.

My intention is that this can be used by my Master and Ph.D. students as the getting started kit for their EEG research.   These algorithms may not necessary work well particularly for EEG emotion recognition, but I have included all of them for the sake of completeness.

---

Note: Before using the tutorials, please create a folder "data" and download preprocessed DEAP dataset and put s01.dat,...,s32.dat inside this "data" folder.  The data folder will be in the same directory as the tutorial.

Python libraries:
1. Python MNE
2. PyTorch
3. NumPy
4. Sckit-Learn
5. SciPy

Tutorials:
1. (01.) Understanding the DEAP dataset
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
     - Using mean to combine (90% acc)
     - Using sum to combine (91% acc)
     - Using last state to combine (90% acc)
   - (12.) CNN1D + LSTM + MultiHead Self Attention 
     - Using mean to combine (92% acc)
     - Using sum to combine (92% acc)
     - Using last state to combine (94% acc)
   - (13.) ChannelNet ( ) Beau
   - (14.) EEGNet ( ) Chaky
   - (15.) Spatial–Temporal Self-Attention CNN ( ) New + Babby + Beau

---

Some possible conclusions:
- LSTM alone is poor.  This is expected because each single sample is simply a single data point.  Since signal is mostly very long (8064 samples in 1 min), thus it is almost impossible for LSTM to understand the relationship between sample 1 (the first second) and sample 8064 (the last 60th second). 
- It is quite obvious that applying CNN1D before LSTM creates drastic improvement, implying the usefulness of CNN1D in smoothening and convoluting the signals into more meaningful representations for LSTM, thus addressing the LSTM long-term dependencies problem.
- It is also quite clear that using spectrograms can well capture both temporal and frequency information, thus the higher accuracy as shown in CNN2D four layers.  Another point about spectrograms is that it has relatively lesser parameters than others, further motivating its use.
- For both CNN1D and CNN2D, maxpooling proves to be very useful, both downsizing the sample hence speed up the training process, and serves as feature selection as well.
- It is quite clear that adding attention on top of CNN1D + LSTM brings slight benefit, adding around 2% extra accuracy.  But this benefit is not certain, as seen in single head self-attention.  This suggests that adding attention should be carefully designed.  In addition, the added accuracy needs to be justified with great amount of increase in parameters.
- Multihead self attention is clearly better compared to only single head attention, which is consistent to what the Transformers paper have suggested.
- It is worth trying different reduction method (sum, mean, last) but the results may not be straightforward.  This is likely the one of the less prioritized hyperparameters to try.

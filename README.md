# EEG Emotion Recognition (DEAP dataset)

*Authors: myself and my Ph.D. student Mr. Akraradet, and Master students Ms. Pranissa, Ms. Chanapa, and Mr. Pongkorn.*

This repository compares different modeling approaches ranging from traditional machine learning algorithms and deep learning algorithms, by using emotion recognition from the benchmark DEAP dataset as the case study.

My intention is that there are just so many researches out there about the DEAP dataset but they can be hardly compared.  Consequently, as a EEG researcher, it is almost impossible to know what architectural decisions should I make.  This is due to the fact that some paper either did not provide the codebase hence not reproducible, or did not clearly specify the hyperparameters used, or just simply due to the obvious fact that even two papers using the same model cannot be directly compared because of differences in hardware and hyperparameters used.  

Thus I want to make a controlled comparision of typical EEG models to create a clear understanding what works or what does not.

My another intention is that this codebase can be used by my Master and Ph.D. students as the getting started kit for their EEG research, since it mostly covers most of the typical EEG models.

---

Note: Before using the tutorials, please create a folder "data" and download preprocessed DEAP dataset and put s01.dat,...,s32.dat inside this "data" folder.  The data folder will be in the same directory as the tutorial.   Also create an empty "models" folder as well.

Python libraries:
1. Python MNE
2. PyTorch
3. NumPy
4. Sckit-Learn
5. SciPy

---

## Docker Prerequisite (Akraradet & Raknatee)

1. Docker
2. Docker-Compose

## How to use

The docker is designed to use with Visual Studio Code with Docker Extension. This way we can attach `visual code` to the docker environment.

Once you `compose` the service, go to the docker tab and find `eeg-emotion`. Right click and select `Attach Visual Studio Code`. Open the `/root/projects/` and have fun coding.

There are 2 types of docker-compose. CPU only and GPU

- CPU
```sh
docker-compose -f docker-compose-cpu.yml up --build -d
```

- GPU
```sh
docker-compose -f docker-compose-gpu.yml up --build -d
```


---

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
- LSTM alone is poor (71%).  This is expected because each single sample is simply a single data point.  Since signal is mostly very long (8064 samples in 1 min), thus it is almost impossible for LSTM to understand the relationship between sample 1 (the first second) and sample 8064 (the last 60th second). 
- It is quite obvious that applying CNN1D before LSTM creates drastic improvement (92%), implying the usefulness of CNN1D in smoothening and convoluting the signals into more meaningful representations for LSTM, thus addressing the LSTM long-term dependencies problem.
- It is also quite clear that using spectrograms (96%) can well capture both temporal and frequency information, thus the higher accuracy as shown in CNN2D four layers.  Another point about spectrograms is that it has relatively lesser parameters than others, further motivating its use.
- It is important to note that for spectrograms, we have tried different window size and overlap size.  What we found is that its effect is quite minimal and thus we decided to report the best accuracy.  We choose the window size to be simply the sampling rate, and the overlapping size as half the sampling rate.
- For both CNN1D and CNN2D, maxpooling proves to be very useful, both downsizing the sample hence speed up the training process, and serves as feature selection as well.  Anyhow, a small caveat we found is that too much maxpooling can leads to poorer result, likely due to too much downsamplings, hence it is useful to find the optimal layers of maxpooling.
- It is quite clear that adding attention on top of CNN1D + LSTM brings slight benefit, adding around 2% extra accuracy (92 + 2%) as seen in multihead self-attention.  But this benefit is not certain, as seen in single head self-attention.  This suggests that adding attention should be carefully designed.  In addition, the added accuracy needs to be justified with great amount of increase in parameters.
- Multihead self attention (94%) is clearly better compared to only single head attention (91%), which is consistent to what the Transformers paper have suggested.
- It is worth trying different reduction method (sum, mean, last) but the results may not be straightforward.  This is likely one of the less prioritized hyperparameters to try.


---

Result DEAP + SVM with mulitple variables

**Segment First**

|                  | 60s            | 30s            | 20s   | 12s   | 5s | 4s| 3s| 2s| 1s| avg |
|------------------|----------------|----------------|-------|-------|----|---|---|---|---|-----|
| DE               | 0.636&pm;0.040 | 0.647&pm;0.038 | 0.653&pm;0.029 | 0.659&pm;0.029 | 0.664&pm;0.019 | 0.669&pm;0.018 | 0.664&pm;0.016 | 0.675&pm;0.014 | 0.677&pm;0.010 | 0.660&pm;0.024 |
| AS (DASM)        | 0.633&pm;0.038 | 0.656&pm;0.036 | 0.651&pm;0.030 | 0.652&pm;0.026 | 0.648&pm;0.018 | 0.653&pm;0.016 | 0.645&pm;0.013 | 0.650&pm;0.012 | 0.645&pm;0.008 | 0.648&pm;0.022 |
| AS (RASM)        | 0.621&pm;0.050 | 0.636&pm;0.041 | 0.628&pm;0.027 | 0.625&pm;0.021 | 0.617&pm;0.008 | 0.615&pm;0.009 | 0.614&pm;0.005 | 0.615&pm;0.004 | 0.614&pm;0.003 | 0.621&pm;0.019 |
| AS (DCAU)        | 0.641&pm;0.046 | 0.651&pm;0.038 | 0.654&pm;0.032 | 0.654&pm;0.028 | 0.648&pm;0.017 | 0.652&pm;0.017 | 0.643&pm;0.014 | 0.648&pm;0.012 | 0.644&pm;0.008 | 0.648&pm;0.024 |
| CN ($PCC_{time}$)| 0.625&pm;0.024 | 0.633&pm;0.025 | 0.635&pm;0.178 | 0.639&pm;0.019 | 0.641&pm;0.014 | 0.676&pm;0.016 | 0.645&pm;0.011 | 0.647&pm;0.010 | 0.634&pm;0.010 | 0.642&pm;0.034 |
| CN ($PCC_{freq}$)| 0.636&pm;0.042 | 0.648&pm;0.046 | 0.648&pm;0.037 | 0.655&pm;0.032 | 0.662&pm;0.021 | 0.642&pm;0.022 | 0.659&pm;0.019 | 0.662&pm;0.014 | 0.657&pm;0.011 | 0.652&pm;0.027 |
| CN (PLV)         | 0.654&pm;0.040 | 0.668&pm;0.043 | 0.676&pm;0.038 | 0.686&pm;0.032 | 0.693&pm;0.021 | 0.659&pm;0.019 | 0.696&pm;0.018 | 0.698&pm;0.015 | 0.688&pm;0.011 | 0.680&pm;0.026 |
| CN (PLI)         | 0.615&pm;0.048 | 0.623&pm;0.039 | 0.619&pm;0.032 | 0.618&pm;0.026 | 0.615&pm;0.016 | 0.696&pm;0.021 | 0.613&pm;0.013 | 0.613&pm;0.011 | 0.611&pm;0.008 | 0.625&pm;0.024 |
| CSP              | 0.840&pm;0.040 | 0.790&pm;0.041 | 0.787&pm;0.036 | 0.790&pm;0.023 | 0.753&pm;0.020 | 0.750&pm;0.016 | 0.745&pm;0.016 | 0.732&pm;0.014 | 0.763&pm;0.009 | 0.772&pm;0.024 |
| avg              | 0.656&pm;0.041 | 0.661&pm;0.039 | 0.661&pm;0.049 | 0.664&pm;0.026 | 0.660&pm;0.017 | 0.668&pm;0.017 | 0.658&pm;0.014 | 0.660&pm;0.012 | 0.659&pm;0.009 |

**Split First**

|                  | 60s            | 30s            | 20s   | 12s   | 5s | 4s| 3s| 2s| 1s| avg|
|------------------|----------------|----------------|-------|-------|----|---|---|---|---|----|
| DE               | 0.580&pm;0.134 | 0.588&pm;0.130 | 0.593&pm;0.120 | 0.590&pm;0.110 | 0.589&pm;0.105 | 0.595&pm;0.104 | 0.587&pm;0.103 | 0.594&pm;0.101 | 0.591&pm;0.096 | 0.590&pm;0.111 |
| AS (DASM)        | 0.575&pm;0.135 | 0.584&pm;0.127 | 0.587&pm;0.125 | 0.586&pm;0.115 | 0.580&pm;0.106 | 0.584&pm;0.107 | 0.580&pm;0.103 | 0.581&pm;0.103 | 0.578&pm;0.990 | 0.582&pm;0.212 |
| AS (RASM)        | 0.564&pm;0.130 | 0.578&pm;0.125 | 0.568&pm;0.123 | 0.567&pm;0.119 | 0.552&pm;0.123 | 0.552&pm;0.122 | 0.553&pm;0.127 | 0.552&pm;0.124 | 0.553&pm;0.127 | 0.560&pm;0.124 |
| AS (DCAU)        | 0.583&pm;0.134 | 0.592&pm;0.126 | 0.591&pm;0.121 | 0.559&pm;0.113 | 0.584&pm;0.104 | 0.585&pm;0.102 | 0.580&pm;0.102 | 0.582&pm;0.102 | 0.578&pm;0.100 | 0.582&pm;0.112 |
| CN ($PCC_{time}$)| 0.564&pm;0.135 | 0.570&pm;0.128 | 0.572&pm;0.124 | 0.573&pm;0.119 | 0.574&pm;0.116 | 0.574&pm;0.114 | 0.573&pm;0.116 | 0.575&pm;0.114 | 0.574&pm;0.110 | 0.572&pm;0.120 |
| CN ($PCC_{freq}$)| 0.578&pm;0.130 | 0.585&pm;0.122 | 0.585&pm;0.116 | 0.585&pm;0.110 | 0.585&pm;0.106 | 0.581&pm;0.102 | 0.581&pm;0.100 | 0.582&pm;0.099 | 0.577&pm;0.096 | 0.582&pm;0.109 |
| CN (PLV)         | 0.591&pm;0.130 | 0.605&pm;0.119 | 0.609&pm;0.114 | 0.611&pm;0.108 | 0.607&pm;0.102 | 0.604&pm;0.100 | 0.603&pm;0.098 | 0.602&pm;0.096 | 0.596&pm;0.092 | 0.603&pm;0.107 |
| CN (PLI)         | 0.564&pm;0.124 | 0.562&pm;0.117 | 0.559&pm;0.113 | 0.560&pm;0.111 | 0.554&pm;0.107 | 0.555&pm;0.106 | 0.553&pm;0.104 | 0.554&pm;0.104 | 0.554&pm;0.103 | 0.557&pm;0.110 |
| CSP              | 0.830&pm;0.074 | 0.758&pm;0.085 | 0.746&pm;0.080 | 0.731&pm;0.079 | 0.704&pm;0.079 | 0.703&pm;0.080 | 0.700&pm;0.081 | 0.681&pm;0.080 | 0.720&pm;0.074 | 0.730&pm;0.079 |
| avg              | 0.603&pm;0.125 | 0.602&pm;0.120 | 0.601&pm;0.115 | 0.596&pm;0.109 | 0.592&pm;0.105 | 0.593&pm;0.104 | 0.590&pm;0.104 | 0.589&pm;0.103 | 0.591&pm;0.199 | 
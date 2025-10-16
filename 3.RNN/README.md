#Doodle classificiation with RNN

[Very simple to follow explanation of RNN](https://www.youtube.com/watch?v=AsNTP8Kwu80) 


## Iteration 1: training only on animals
- Data: only 10 animals
- Encoding: [dx, dy, pen_lift]; captures motion and stroke boundaries.
- Length rules: drop <6 steps, cap 250; stable, faster training.
- Collation: pad with lengths; pack sequences so GRU ignores padding.
- Model: 2-layer bidirectional GRU (hidden 192); accuracy-speed balance.
- Optimization: AdamW, label smoothing, dropout, grad clipping; robustness.
- Scheduling: ReduceLROnPlateau + early stopping; efficient convergence.
- System: Apple MPS acceleration; save best/last checkpoints for deployment.

## Results & Plots
Test accuracy: 0.9788
Dev accuracy: 0.9945

Epoch 15 | train_loss=0.3095 train_acc1=1.000 val_loss=0.3140 val_acc1=0.909 val_acc3=0.983 | 114.9s


<img src="archive/plots/rnn_confusion_matrix.png" alt="Confusion matrix" width="420"/>

<img src="archive/plots/rnn_per_class_accuracy.png" alt="Per-class accuracy" width="420"/>

<img src="archive/plots/rnn_reliability.png" alt="Reliability diagram" width="420"/>

<img src="archive/plots/rnn_confidence_hist.png" alt="Confidence histogram" width="420"/>


Detected 10-class train split: /Users/elliot18/Desktop/Home/Projects/Digit-Classifier-NN-from-scratch/3.RNN/archive/animal_doodles_10_train.csv
Using device: mps
Classes (10): ['butterfly', 'cow', 'elephant', 'giraffe', 'monkey', 'octopus', 'scorpion', 'shark', 'snake', 'spider']
Train samples: 21675, Val samples: 3825
Loaded sequence cache (train) from /Users/elliot18/Desktop/Home/Projects/Digit-Classifier-NN-from-scratch/3.RNN/archive/seq_cache_v1
Loaded sequence cache (val) from /Users/elliot18/Desktop/Home/Projects/Digit-Classifier-NN-from-scratch/3.RNN/archive/seq_cache_v1
Resumed weights from /Users/elliot18/Desktop/Home/Projects/Digit-Classifier-NN-from-scratch/3.RNN/archive/rnn_animals_best.pt
  Batch 100/339 | loss=1.6417 acc1=0.469 lr=0.00102
  Batch 200/339 | loss=1.3687 acc1=0.547 lr=0.00108
  Batch 300/339 | loss=1.1705 acc1=0.625 lr=0.00119
Epoch 01 | train_loss=1.3891 train_acc1=0.555 val_loss=0.9104 val_acc1=0.694 val_acc3=0.924 | 113.2s
  Batch 100/339 | loss=1.0126 acc1=0.688 lr=0.00138
  Batch 200/339 | loss=0.9124 acc1=0.750 lr=0.00156
  Batch 300/339 | loss=0.8187 acc1=0.781 lr=0.00175
Epoch 02 | train_loss=0.9580 train_acc1=0.736 val_loss=0.6504 val_acc1=0.789 val_acc3=0.955 | 189.6s
  Batch 100/339 | loss=0.7398 acc1=0.812 lr=0.00203
  Batch 200/339 | loss=0.7064 acc1=0.859 lr=0.00224
  Batch 300/339 | loss=0.7100 acc1=0.875 lr=0.00243
Epoch 03 | train_loss=0.8041 train_acc1=0.805 val_loss=0.5446 val_acc1=0.812 val_acc3=0.964 | 86.3s
  Batch 100/339 | loss=0.6413 acc1=0.828 lr=0.00267
  Batch 200/339 | loss=0.5979 acc1=0.922 lr=0.00281
  Batch 300/339 | loss=0.6871 acc1=0.844 lr=0.00291
Epoch 04 | train_loss=0.7026 train_acc1=0.848 val_loss=0.4482 val_acc1=0.854 val_acc3=0.972 | 71.9s
  Batch 100/339 | loss=0.5341 acc1=0.891 lr=0.00299
  Batch 200/339 | loss=0.6036 acc1=0.938 lr=0.003
  Batch 300/339 | loss=0.5521 acc1=0.906 lr=0.00299
Epoch 05 | train_loss=0.6231 train_acc1=0.882 val_loss=0.4408 val_acc1=0.855 val_acc3=0.973 | 114.5s
  Batch 100/339 | loss=0.5256 acc1=0.938 lr=0.00296
  Batch 200/339 | loss=0.5237 acc1=0.906 lr=0.00292
  Batch 300/339 | loss=0.4929 acc1=0.938 lr=0.00288
Epoch 06 | train_loss=0.5532 train_acc1=0.911 val_loss=0.3922 val_acc1=0.876 val_acc3=0.979 | 95.3s
  Batch 100/339 | loss=0.4784 acc1=0.938 lr=0.0028
  Batch 200/339 | loss=0.5114 acc1=0.953 lr=0.00272
  Batch 300/339 | loss=0.4235 acc1=0.969 lr=0.00265
Epoch 07 | train_loss=0.4852 train_acc1=0.940 val_loss=0.3912 val_acc1=0.878 val_acc3=0.977 | 84.4s
  Batch 100/339 | loss=0.4140 acc1=0.969 lr=0.00252
  Batch 200/339 | loss=0.5013 acc1=0.938 lr=0.00242
  Batch 300/339 | loss=0.3819 acc1=0.984 lr=0.00232
Epoch 08 | train_loss=0.4384 train_acc1=0.960 val_loss=0.4043 val_acc1=0.876 val_acc3=0.978 | 98.4s
  Batch 100/339 | loss=0.3950 acc1=0.984 lr=0.00216
  Batch 200/339 | loss=0.3761 acc1=0.984 lr=0.00204
  Batch 300/339 | loss=0.3777 acc1=1.000 lr=0.00192
Epoch 09 | train_loss=0.3988 train_acc1=0.977 val_loss=0.3727 val_acc1=0.890 val_acc3=0.978 | 124.0s
  Batch 100/339 | loss=0.3623 acc1=0.984 lr=0.00175
  Batch 200/339 | loss=0.4179 acc1=0.953 lr=0.00162
  Batch 300/339 | loss=0.3546 acc1=1.000 lr=0.00149
Epoch 10 | train_loss=0.3667 train_acc1=0.989 val_loss=0.3383 val_acc1=0.901 val_acc3=0.979 | 116.1s
  Batch 100/339 | loss=0.3328 acc1=1.000 lr=0.00131
  Batch 200/339 | loss=0.3509 acc1=0.984 lr=0.00119
  Batch 300/339 | loss=0.3272 acc1=1.000 lr=0.00107
Epoch 11 | train_loss=0.3394 train_acc1=0.997 val_loss=0.3125 val_acc1=0.909 val_acc3=0.983 | 106.2s
  Batch 100/339 | loss=0.3270 acc1=1.000 lr=0.000902
  Batch 200/339 | loss=0.3214 acc1=1.000 lr=0.00079
  Batch 300/339 | loss=0.3127 acc1=1.000 lr=0.000684
Epoch 12 | train_loss=0.3236 train_acc1=0.999 val_loss=0.3144 val_acc1=0.909 val_acc3=0.981 | 90.8s
  Batch 100/339 | loss=0.3135 acc1=1.000 lr=0.000549
  Batch 200/339 | loss=0.3152 acc1=1.000 lr=0.00046
  Batch 300/339 | loss=0.3089 acc1=1.000 lr=0.00038
Epoch 13 | train_loss=0.3158 train_acc1=1.000 val_loss=0.3148 val_acc1=0.911 val_acc3=0.982 | 70.9s
  Batch 100/339 | loss=0.3109 acc1=1.000 lr=0.000284
  Batch 200/339 | loss=0.3146 acc1=1.000 lr=0.000227
  Batch 300/339 | loss=0.3102 acc1=1.000 lr=0.00018
Epoch 14 | train_loss=0.3117 train_acc1=1.000 val_loss=0.3144 val_acc1=0.909 val_acc3=0.982 | 75.8s
  Batch 100/339 | loss=0.3081 acc1=1.000 lr=0.000132
  Batch 200/339 | loss=0.3082 acc1=1.000 lr=0.000111
  Batch 300/339 | loss=0.3070 acc1=1.000 lr=0.000101
Epoch 15 | train_loss=0.3095 train_acc1=1.000 val_loss=0.3140 val_acc1=0.909 val_acc3=0.983 | 114.9s
Early stopping triggered.
Saved model to: /Users/elliot18/Desktop/Home/Projects/Digit-Classifier-NN-from-scratch/3.RNN/archive/rnn_animals_last.pt
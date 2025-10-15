#Doodle classificiation with RNN

[Very simple to follow explanation of RNN](https://www.youtube.com/watch?v=AsNTP8Kwu80) 


## Iteration 1: training only on animals
- Data: animal_doodles.csv, recognized only; cleaner labels and training.
- Encoding: [dx, dy, pen_lift]; captures motion and stroke boundaries.
- Length rules: drop <6 steps, cap 250; stable, faster training.
- Collation: pad with lengths; pack sequences so GRU ignores padding.
- Model: 2-layer bidirectional GRU (hidden 192); accuracy-speed balance.
- Optimization: AdamW, label smoothing, dropout, grad clipping; robustness.
- Scheduling: ReduceLROnPlateau + early stopping; efficient convergence.
- System: Apple MPS acceleration; save best/last checkpoints for deployment.

## 1st Iteration results

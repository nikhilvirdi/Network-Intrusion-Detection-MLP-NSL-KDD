## Network Intrusion Detection — MLP

A multi-layer perceptron built in PyTorch to classify network connection records as normal or one of several attack categories, using the NSL-KDD dataset (125,973 training records, 22,544 test records, 41 features).

Built as a hands-on project to learn core deep learning fundamentals: the forward/backward pass, loss functions and optimizers (cross-entropy, Adam), and overfitting and regularization. NSL-KDD's test set is deliberately harder than its training set, so there's a real generalization gap to observe here rather than a hypothetical one.

### Stack

- PyTorch
- Google Colab
- pandas / scikit-learn (preprocessing)
- matplotlib / seaborn (visualization)

### Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) is a refined, de-duplicated version of the original KDD Cup 99 dataset. Each record is a summarized network connection with 41 features (38 numeric, 3 categorical: `protocol_type`, `service`, `flag`) and a label naming the specific attack, or `normal`.

The training set contains 23 distinct labels. The test set contains 38, including 17 attack types with zero training examples. That gap is the point of using this dataset: a model can only do well on those 17 by learning general attack patterns, not by memorizing anything it's already seen.

### Problem framing

23 raw attack names is too granular to learn well (several have single-digit example counts); binary normal-vs-attack throws away information about attack type. This project uses the standard 5-class split instead: `normal`, plus the four established attack families — `DoS`, `Probe`, `R2L`, `U2R`.

### Pipeline

1. Load raw NSL-KDD files, attach column names (the source files ship with no header row).
2. Explore the data: label distribution, train/test category overlap, categorical cardinality, numeric feature ranges.
3. One-hot encode the 3 categorical columns (84 resulting columns), scale the 38 numeric columns with `StandardScaler`. Fit only on training data throughout, to avoid leaking test-set information into preprocessing.
4. Map the 23 raw attack labels into the 5-class scheme; encode as integers.
5. Baseline: majority-class guess (43.1% accuracy) and Logistic Regression (76.1%), as reference points for the MLP.
6. MLP: 122 inputs → 128 → 64 → 5, ReLU activations, trained with `CrossEntropyLoss` and Adam for 10 epochs.
7. Diagnose class imbalance via per-class precision/recall and a confusion matrix.
8. Address it: tried class-weighted loss (two variants) and oversampling (two variants); kept the best-performing one.
9. Save the final model, run inference on individual sample rows.

### Results

Final model: 76% overall test accuracy, matching plain Logistic Regression on that headline number, but with a distinct per-class shape:

| Class  | Precision | Recall |
|--------|-----------|--------|
| DoS    | 96%       | 82%    |
| Probe  | 70%       | 68%    |
| R2L    | 97%       | 9%     |
| U2R    | 64%       | 31%    |
| normal | 67%       | 94%    |

DoS and normal are handled well. U2R improved substantially through oversampling (recall roughly doubled from its original 15%). R2L is the project's clearest open problem: when the model does flag it, it's almost always right, but it misses the large majority of real R2L attacks. Every technique tried — class weighting at two strengths, oversampling at two scales — left this mostly unchanged. Several of R2L's test-time attack types were never seen in training at all, which looks like the actual cause, rather than anything a different loss or resampling ratio would fix.

### What this project actually showed

Overall accuracy alone hid a real failure: 76% looked respectable until per-class metrics showed R2L and U2R were barely being detected at all. Training loss dropping to near-zero didn't mean the model generalized — test loss sat nearly 300x higher than training loss on the first run, direct evidence of overfitting rather than genuine learning. And fixing class imbalance had no single right answer: the same technique (aggressive weighting) that helped R2L's recall wrecked U2R's precision, and oversampling that worked cleanly for one rare class destabilized things when applied to two at once.

### Running it

Open `notebooks/network_intrusion_detection_mlp.ipynb` in Google Colab and run top to bottom. Every step includes markdown explaining what it does and why, before the corresponding code cell.

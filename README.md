## Network Intrusion Detection MLP (NSL-KDD Dataset)

<img width="300" alt="nslkdddlogo" src="https://github.com/user-attachments/assets/8aad9a49-fb66-4b9b-a55a-13a61a6ff448" />


A multi-layer perceptron built in PyTorch to classify network connection records as normal or one of several attack categories, using the NSL-KDD dataset (125,973 training records, 22,544 test records, 41 features).

Built as a hands-on project to learn core deep learning fundamentals: the forward/backward pass, loss functions and optimizers (cross-entropy, Adam), and overfitting and regularization. NSL-KDD's test set is deliberately harder than its training set, so there's a real generalization gap to observe here rather than a hypothetical one.

### Stack

- PyTorch
- Google Colab
- pandas / scikit-learn (preprocessing)
- matplotlib / seaborn (visualization)

### Two notebook versions

This repo keeps both stages of the project, rather than only the final result, since the progression is part of what the project demonstrates:

- **`notebooks/network_intrusion_detection_mlp_initial.ipynb`** — the base pipeline and model, no regularization. 76% accuracy.
- **`notebooks/network_intrusion_detection_mlp_enhanced.ipynb`** — everything in the initial notebook, plus Dropout, BatchNorm, L2 weight decay, and early stopping. 78% accuracy.

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
6. MLP: 122 inputs → 128 → 64 → 5, ReLU activations, trained with `CrossEntropyLoss` and Adam.
7. Diagnose class imbalance via per-class precision/recall and a confusion matrix.
8. Address it: tried class-weighted loss (two variants) and oversampling (two variants); kept the best-performing one. *(Initial notebook ends here.)*
9. Add Dropout, BatchNorm, and L2 weight decay to the architecture and optimizer; retrain with early stopping instead of a fixed epoch count.
10. Test a learning rate scheduler and L1 regularization; keep what helps, document and discard what doesn't.
11. Save the final model, run inference on individual sample rows.

### Results

| Class  | Initial (P/R) | Enhanced (P/R) |
|--------|----------------|-----------------|
| DoS    | 96% / 82%      | 96% / 83%       |
| Probe  | 70% / 68%      | 81% / 69%       |
| R2L    | 97% / 9%       | 96% / 10%       |
| U2R    | 64% / 31%      | 72% / 43%       |
| normal | 67% / 94%      | 68% / 96%       |
| **Overall accuracy** | **76%** | **78%** |

DoS and normal are handled well in both versions. U2R improved substantially through oversampling in the initial model (recall roughly doubled from its original 15%), then improved further in the enhanced model through regularization. Probe also firmed up noticeably in the enhanced version, gaining 11 points of precision with no loss in recall.

R2L is the project's clearest, most persistent problem: when the model does flag it, it's almost always right, but it misses the large majority of real R2L attacks. Every technique tried — class weighting at two strengths, oversampling at two scales, and every regularization technique in the enhanced model — left this mostly unchanged. Several of R2L's test-time attack types were never seen in training at all, which looks like the actual cause, rather than anything a different loss, resampling ratio, or regularization setup would fix.

### What this project actually showed

Overall accuracy alone hid a real failure: 76% looked respectable until per-class metrics showed R2L and U2R were barely being detected at all. Training loss dropping to near-zero didn't mean the model generalized — test loss sat nearly 300x higher than training loss on the first run, direct evidence of overfitting rather than genuine learning.

Fixing class imbalance had no single right answer: the same technique (aggressive weighting) that helped R2L's recall wrecked U2R's precision, and oversampling that worked cleanly for one rare class destabilized things when applied to two at once.

Regularization (Dropout, BatchNorm, L2, early stopping) closed a real chunk of the overfitting gap and improved Probe and U2R meaningfully, without hurting DoS or normal — but it didn't touch R2L at all, confirming that overfitting and data scarcity are different problems needing different fixes. Not every technique tried was kept: a learning rate scheduler never got the chance to matter, and L1 regularization was tested and rejected after it collapsed U2R's precision, despite looking marginally better on loss alone.

### Running it

Open either notebook in Google Colab and run top to bottom. Every step includes markdown explaining what it does and why, before the corresponding code cell.

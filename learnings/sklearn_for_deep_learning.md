## scikit-learn for Deep Learning

scikit-learn doesn't build neural networks (that's PyTorch's job), but almost
every deep learning project leans on it before and after the network itself:
splitting data, preprocessing features, running a quick baseline, and scoring
results honestly. This file covers the parts of scikit-learn that show up
repeatedly in that role, with runnable code and real output for each.

Every example below uses one consistent synthetic dataset: 300 network
connection records with `duration`, `src_bytes`, `dst_bytes` (numeric),
`protocol_type`, `flag` (categorical), and a `label` (`normal`, `DoS`,
`Probe`) — the same shape of problem as a typical intrusion-detection or
tabular classification project, so the examples read the same way regardless
of which specific project you're using this file for.

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 300
df = pd.DataFrame({
    'duration': np.random.exponential(50, n),
    'src_bytes': np.random.exponential(500, n),
    'dst_bytes': np.random.exponential(300, n),
    'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n),
    'flag': np.random.choice(['SF', 'S0', 'REJ'], n),
    'label': np.random.choice(['normal', 'DoS', 'Probe'], n, p=[0.6, 0.3, 0.1])
})
print(df.shape)   # (300, 6)
print(df.head(3))
#      duration   src_bytes  dst_bytes protocol_type flag   label
# 0   23.463404   26.532548  55.514203          icmp  REJ  normal
# 1  150.506072  378.954470  97.964436           tcp   S0     DoS
# 2   65.837285  388.955221  58.443545          icmp   SF     DoS
```

### Splitting Data: `train_test_split`

**Theory:** a model must be evaluated on data it never saw during training or
tuning — otherwise the evaluation number reflects memorization, not
generalization. `train_test_split` does one random split; for classification,
always pass `stratify=y` so each class's proportion is preserved in both
halves, not skewed by chance.

```python
from sklearn.model_selection import train_test_split

X_raw = df.drop(columns=['label'])
y_raw = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)
print(X_train.shape, X_test.shape)   # (240, 5) (60, 5)
```

- `test_size=0.2` — 20% held out for testing, 80% for training. `0.2`-`0.3` is
  a common range; smaller test sets give noisier evaluation, larger ones
  leave less data to train on.
- `random_state=42` — fixes which rows land in which split, so re-running
  gives the identical split every time.
- `stratify=y_raw` — without this, a rare class could end up almost entirely
  in one split by pure chance, especially with small or imbalanced data.

**Use case in DL specifically:** many DL projects use an existing, pre-split
train/test file (like NSL-KDD's own train/test files) rather than calling
`train_test_split` directly — in that case, this function is still used to
carve a **validation set** out of the training data, so the test set stays
untouched until final evaluation.

### Preprocessing: `StandardScaler`

**Theory:** neural networks are sensitive to input scale — a feature ranging
into the thousands sitting beside one bounded 0-1 can dominate a network's
early computations purely because its numbers are bigger, unrelated to how
informative it actually is. `StandardScaler` rescales each feature to zero
mean, unit standard deviation: `(x - mean) / std`.

```python
from sklearn.preprocessing import StandardScaler

num_cols = ['duration', 'src_bytes', 'dst_bytes']
scaler = StandardScaler()
scaler.fit(X_train[num_cols])
scaled_train = scaler.transform(X_train[num_cols])

print(round(scaled_train[:, 0].mean(), 6))   # -0.0  (floating-point zero)
print(round(scaled_train[:, 0].std(), 6))    # 1.0
```

**The fit/transform pattern, which repeats across nearly every scikit-learn
preprocessing tool:**
- `.fit(data)` — learns parameters from data (here: mean and std per column).
  Doesn't change anything, just records numbers.
- `.transform(data)` — applies those already-learned parameters to data.
- **Always `fit` on training data only**, then `transform` both train and
  test with that same fitted object. Fitting on test data leaks information
  about the test distribution into preprocessing, quietly inflating how good
  your evaluation looks.

**Alternatives:** `MinMaxScaler` squeezes values into a fixed range (usually
0-1) based on the data's exact min/max — more sensitive to outliers than
`StandardScaler`, since one extreme value stretches the whole scale.
`RobustScaler` uses median and interquartile range instead of mean/std,
tolerating outliers better than either.

### Encoding Categories: `OneHotEncoder` and `LabelEncoder`

**Theory:** a neural network's input layer only does matrix multiplication —
it cannot consume the literal text `'tcp'`. Categorical text columns need
converting to numbers first, and *how* depends on whether the column is an
**input feature** or the **target label**.

**`OneHotEncoder`** — for input features. Each unique category becomes its
own 0/1 column, so no artificial ordering is implied between categories that
have none (`'tcp'` isn't "more" than `'udp'`).

```python
from sklearn.preprocessing import OneHotEncoder

cat_cols = ['protocol_type', 'flag']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(X_train[cat_cols])
encoded_train = encoder.transform(X_train[cat_cols])

print(encoded_train.shape)     # (240, 6)  — 3 protocol_type + 3 flag values
print(encoder.categories_)
# [array(['icmp', 'tcp', 'udp'], dtype=object),
#  array(['REJ', 'S0', 'SF'], dtype=object)]
```

- `sparse_output=False` — returns a dense NumPy array (every 0 explicitly
  stored) rather than a memory-efficient sparse matrix; dense is easier to
  inspect directly, and fine for small-to-moderate column counts.
- `handle_unknown='ignore'` — if `.transform()` later meets a category never
  seen during `.fit()`, it outputs all-zeros for that row's group instead of
  raising an error. Without this, an unseen category at test time crashes.

**`LabelEncoder`** — for the target column only. Assigns each class a single
integer (alphabetically, by default) rather than expanding into multiple
columns, since a classifier's output layer predicts one class index, not
several parallel yes/no columns.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(y_train)
y_train_enc = le.transform(y_train)

print(le.classes_)             # ['DoS' 'Probe' 'normal']
print(y_train_enc[:5])         # [2 0 2 2 2]
```

**Gotcha — using `LabelEncoder` on an input feature instead of `OneHotEncoder`:**
this silently implies a false ordering. Encoding `protocol_type` as
`{'icmp': 0, 'tcp': 1, 'udp': 2}` tells the model `'udp'` is somehow "twice"
`'tcp'` — nonsense for an unordered category, and a model can pick up on
that fake numeric relationship as if it were real signal. Reserve
`LabelEncoder` for targets; use `OneHotEncoder` for unordered categorical
inputs.

### Class Imbalance: `compute_class_weight`

**Theory:** when one class vastly outnumbers another in training data, a
model can achieve deceptively high accuracy by mostly ignoring the rare
class. `compute_class_weight` computes a per-class multiplier — inversely
proportional to how frequent each class is — that can be fed into a loss
function to penalize mistakes on rare classes more heavily.

```python
from sklearn.utils.class_weight import compute_class_weight

weights = compute_class_weight('balanced', classes=np.unique(y_train_enc), y=y_train_enc)
print(dict(zip(le.classes_, weights.round(3))))
# {'DoS': 0.988, 'Probe': 3.478, 'normal': 0.588}
```

`'balanced'` uses the formula `total_samples / (num_classes × class_count)`
— note `Probe` (the rarest class here) gets the highest weight. In PyTorch,
this array gets passed directly into `nn.CrossEntropyLoss(weight=...)`.

**Debugging tip:** extremely skewed weights (hundreds or thousands×) can
overcorrect a model into false-positive-heavy predictions for the rare
class. If precision on that class collapses after weighting, try dampening
the weights (e.g. a square root) rather than abandoning the technique
outright, or consider oversampling the rare class's real training rows
instead of just reweighting the loss.

### Pipelines and Column-Specific Preprocessing: `Pipeline`, `ColumnTransformer`

**Theory:** applying scaling to numeric columns and encoding to categorical
columns separately, then manually stitching the results back together, works
but is easy to get subtly wrong (mismatched row order, forgetting to refit
consistently). `ColumnTransformer` applies different preprocessing to
different column subsets in one object; `Pipeline` chains preprocessing and
a model together so `.fit()`/`.predict()` run the whole sequence in one call.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('clf', LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train_enc)
preds = pipe.predict(X_test)
```

**Why this matters beyond convenience:** a `Pipeline` used inside cross-
validation (below) automatically refits preprocessing on only each fold's
training portion — preventing the exact leakage mistake `StandardScaler`'s
section warned about, without you having to remember to do it manually every
time.

### Baseline Models: `LogisticRegression`, `DummyClassifier`

**Theory:** before trusting a neural network's accuracy number, compare it
against something far simpler. If a plain baseline scores nearly as well,
that's real information — either the added complexity isn't earning its
keep, or something upstream (features, preprocessing, training) isn't
working as intended.

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train_enc)
dummy_preds = dummy.predict(X_test)

y_test_enc = le.transform(y_test)
print(round(accuracy_score(y_test_enc, dummy_preds), 4))   # 0.5667
print(round(accuracy_score(y_test_enc, preds), 4))          # 0.55  (LogisticRegression, via pipe)
```

- `DummyClassifier(strategy='most_frequent')` — always predicts the single
  most common training class. This is the true floor: any real model must
  clearly beat it, or it's adding no value.
- `LogisticRegression` — a genuine, if simple, classifier; a normal first
  real baseline before deep learning. On this random synthetic data both
  scores land close together, which is itself expected — there's no real
  signal in randomly generated labels for either model to find.

### Evaluation Beyond Accuracy: `classification_report`, `confusion_matrix`

**Theory:** overall accuracy can hide serious per-class failure, especially
with imbalanced classes. **Precision** (of everything predicted as class X,
what fraction actually was X) and **recall** (of everything that actually
was X, what fraction got caught) tell a fuller story per class.
`confusion_matrix` shows exactly what got predicted instead of the right
answer, not just a right/wrong count.

```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test_enc, preds, target_names=le.classes_, zero_division=0))
#               precision    recall  f1-score   support
#          DoS       0.00      0.00      0.00        20
#        Probe       0.00      0.00      0.00         6
#       normal       0.57      0.97      0.72        34
#     accuracy                           0.55        60
#    macro avg       0.19      0.32      0.24        60
# weighted avg       0.32      0.55      0.41        60

print(confusion_matrix(y_test_enc, preds))
# [[ 0  0 20]
#  [ 1  0  5]
#  [ 1  0 33]]
```

- `zero_division=0` — suppresses a warning (and avoids a `nan`) when a class
  has zero predicted or zero true instances, common early in a project when
  a model hasn't learned a rare class yet.
- **macro avg vs weighted avg** — macro avg treats every class equally
  regardless of size; weighted avg lets large classes dominate the average.
  A big gap between the two is itself a signal that performance is uneven
  across classes.
- Reading the confusion matrix: row = true class, column = predicted class.
  Here, all 20 true `DoS` rows landed in the `normal` column — the model
  defaulted to the majority class whenever uncertain, visible directly in
  the matrix rather than inferred from the report alone.

### Cross-Validation and Hyperparameter Search: `cross_val_score`, `GridSearchCV`

**Theory:** a single train/test split gives one noisy estimate of
performance. Cross-validation splits data into several folds, trains and
evaluates on each fold rotation, and reports the spread — a more reliable
estimate, and a way to compare hyperparameter settings without repeatedly
touching the real held-out test set.

```python
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

cv_scores = cross_val_score(pipe, X_raw, le.transform(y_raw), cv=KFold(n_splits=5, shuffle=True, random_state=42))
print(cv_scores.round(3))          # [0.533 0.6   0.533 0.517 0.533]
print(round(cv_scores.mean(), 4))  # 0.5433

param_grid = {'clf__C': [0.1, 1.0, 10.0]}
grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(X_train, y_train_enc)
print(grid.best_params_)           # {'clf__C': 0.1}
print(round(grid.best_score_, 4))  # 0.5625
```

- `'clf__C'` — the double underscore addresses a parameter (`C`, Logistic
  Regression's regularization strength) inside a named pipeline step
  (`clf`). This naming pattern is required whenever tuning parameters of a
  step nested inside a `Pipeline`.
- **Use case in DL:** `GridSearchCV` itself is rarely used to tune a neural
  network directly (training loops don't fit scikit-learn's `.fit()`
  interface cleanly without a wrapper) — but it's exactly the right tool for
  tuning the classical baseline model, or for tuning preprocessing choices
  upstream of the network.

### Common Mistakes and Debugging Tips

**Fitting a scaler/encoder on combined train+test data:**

```python
leaky_scaler = StandardScaler()
leaky_scaler.fit(pd.concat([X_train[num_cols], X_test[num_cols]]))   # WRONG — leaks test info
correct_scaler = StandardScaler()
correct_scaler.fit(X_train[num_cols])                                # correct

print(not np.allclose(leaky_scaler.mean_, correct_scaler.mean_))   # True — they really differ
```
This is the single most common, hardest-to-notice mistake in this whole
area: it doesn't crash, doesn't warn, and quietly makes test-set evaluation
look better than a model would actually perform on genuinely new data.

**Passing an unseen category without `handle_unknown='ignore'`:**

```python
strict_encoder = OneHotEncoder(handle_unknown='error')   # default behavior
strict_encoder.fit(X_train[cat_cols])
# strict_encoder.transform(pd.DataFrame({'protocol_type': ['sctp'], 'flag': ['SF']}))
# ValueError: Found unknown categories ['sctp'] in column 0 during transform
```
Real-world or test data can contain a category never seen in training.
`handle_unknown='ignore'` prevents an outright crash; the resulting all-zero
row still needs to be accounted for in how the model interprets it.

**Using `LabelEncoder` for a multi-column categorical feature:** it only
accepts a single 1-D array — calling it on a full categorical DataFrame
column-by-column, encoding each independently, is fine; calling it expecting
multi-column one-hot-style behavior is a conceptual mix-up with
`OneHotEncoder`, not something `LabelEncoder` is built for.

**Forgetting `stratify=y` on an imbalanced dataset:** a random split can, by
chance, put most or all of a rare class into one side of the split —
sometimes leaving zero examples of it in training, or in validation, purely
from bad luck rather than a real data problem.

### Quick Recap

```text
scikit-learn's role in a DL project: preprocessing before the network,
baselines and honest scoring after it — not building the network itself.

train_test_split(..., stratify=y)     — honest split, preserves class ratios
StandardScaler                        — zero mean, unit variance; fit on train only
OneHotEncoder                         — unordered categorical inputs → 0/1 columns
LabelEncoder                          — target classes → single integer, never for inputs
compute_class_weight('balanced', ...) — per-class loss multiplier for imbalance
ColumnTransformer + Pipeline          — different preprocessing per column type, one object
DummyClassifier / LogisticRegression  — the floor and the real bar before deep learning
classification_report / confusion_matrix — per-class truth, not just overall accuracy
cross_val_score / GridSearchCV        — reliable estimates and tuning without touching test data

Golden rule: fit every preprocessing object on training data only, transform
everything else with that same fitted object — the single mistake that most
often makes an evaluation number lie to you without any error or warning.
```

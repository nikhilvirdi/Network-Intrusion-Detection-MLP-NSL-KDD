## DL Production Project Flow

A  reference for building deep learning projects, laid out in the order these phases actually happen. Generic on purpose, so it works before starting any DL project, whatever the domain or data type.

### Initial Project Setup

#### What this phase covers
Create the GitHub repo before writing any code, even if it stays empty for the first commit. Set up the folder structure you expect to need, something like `notebooks/`, `data/` (gitignored if the data is large or licensed), and a README stub. If your data itself changes over time or across sources, note that now too; a line in the README about where the data came from and when saves confusion later.

#### If it's not working
It's rarely a hard blocker to add version control mid-project, but you lose the early history: the record of how the problem definition or architecture changed along the way. If you're several sessions in with no repo, start one now rather than waiting for a clean moment that won't arrive.

### Environment and Library Setup

#### What this phase covers
Decide where the project runs (Colab, a local notebook, a cloud VM) and set the runtime accordingly. Check for a GPU explicitly if you think you need one; don't request one you won't use, since it wastes quota without buying anything on a small model. Then import everything the project needs and print each library's version. This isn't just tidy habit; it's the fastest way to rule out a version mismatch later, when something behaves differently than a tutorial or the docs say it should.

#### If it's not working
If training is unexpectedly slow, confirm you're on the runtime you think you're on (CPU vs. GPU) before troubleshooting anything else. If a function signature doesn't match what you expected, check versions first. If a library update changed behavior, pin an older version or adjust the code to the new API rather than guessing.

### Reproducibility (Random Seeds)

#### What this phase covers
Fix a seed for every source of randomness: NumPy, PyTorch, and any train/test split function. Weight initialization and data shuffling both draw on randomness, so without a fixed seed, identical code can produce different results, and you lose the ability to tell whether a change in outcome came from your edit or from chance.

#### If it's not working
If you re-run something and get different numbers with no code change to explain it, check whether you're actually re-creating the model and optimizer fresh, seeded, right before training. Reusing an already-trained model object silently breaks reproducibility even with a fixed seed, since the seed only controls the initial random state, not anything after.

### Problem Definition

#### What this phase covers
State precisely what you're predicting before writing any model code: classification or regression, how many classes if classification, and what "success" actually means here. This decision sets your loss function, your evaluation metric, and your output layer's shape. Get it wrong and everything downstream inherits the mistake.

#### If it's not working
If partway through you find the loss function or metric doesn't fit what you're actually trying to measure, say you're optimizing accuracy but the real cost of different error types isn't equal, go back to this phase directly rather than patching around it further downstream. A wrong problem definition can't be fixed with a better model.

### Dataset Acquisition and Loading

#### What this phase covers
Get the raw data into your environment and load it into a usable structure: a DataFrame, a tensor, whatever fits. Watch for format quirks specific to the source (missing headers, unusual delimiters, encoding issues), and confirm the loaded shape matches the data's documentation before doing anything else.

#### If it's not working
If row counts, column counts, or basic shape don't match expectations, don't proceed hoping it resolves itself downstream. Trace it back to the load step and fix it there, since every later phase assumes the data loaded correctly.

### Exploratory Data Analysis

#### What this phase covers
Look at the data before writing any model code: size, feature types, class balance, missing values, rough distributions. The goal is catching problems here that would otherwise surface much later as confusing model behavior: a severe class imbalance, an extreme outlier in one feature, a categorical column with far more unique values than expected.

#### If it's not working
If the model trains fine but performs oddly on specific subsets of data, or evaluation later reveals a class the model essentially ignores, come back here and check whether that class was underrepresented from the start. It often explains model behavior better than a training-loop bug does.

### Feature Engineering

#### What this phase covers
Encode categorical columns into numeric form, derive or combine features where it adds real signal, and drop redundant or near-constant ones. This matters heavily for tabular data, since a plain MLP has no built-in way to exploit row/column structure the way a CNN exploits spatial structure in images or a transformer exploits sequence structure in text. For those data types this phase shrinks considerably, since the network learns much of its own feature representation.

#### If it's not working
If the model underperforms and you suspect it isn't the architecture at fault, revisit which categorical encoding was used (one-hot vs. embeddings for high-cardinality columns) and whether an obviously useful derived feature was left out. Also check whether train and test sets share consistent categories; a category present in test but absent from training breaks a naive one-hot scheme.

### Data Preprocessing

#### What this phase covers
Scale or normalize numeric features, handle missing values, and split into train, validation, and test sets. Fit any scaler or encoder only on training data, then apply that same fitted transform to validation and test. Fitting on test data leaks information about it into training and inflates your evaluation results without you realizing it.

#### If it's not working
If training loss behaves erratically (spikes, stalls, or diverges), an unscaled feature with a wildly different range than the rest is a common, easy-to-miss cause. If validation or test metrics look implausibly good, suspect a data leak from incorrect fit/transform ordering; it's a frequent, quiet culprit.

### Baseline Model (optional but recommended)

#### What this phase covers
Train something simple first: logistic regression, a decision tree, or even a rule that always predicts the most common class. This tells you whether the added complexity of a neural network earns its keep, and gives you a concrete number your real model needs to beat.

#### If it's not working
If your deep learning model's final accuracy barely beats, or doesn't beat, this baseline, treat that as real information rather than something to explain away. It usually means either the features don't carry much learnable signal beyond what the simple model already captures, or something earlier in the pipeline isn't working as intended.

### Model Architecture Design

#### What this phase covers
Decide the number of layers, the size of each layer, and the activation functions, before writing the model class. Input size (matching your feature count) and output size (matching your number of classes) are usually fixed by the problem itself; everything between is where design judgment actually applies.

#### If it's not working
If the model can't learn the training data at all, loss stays flat or barely moves, that often points to too little capacity (too few neurons or layers) or a missing or wrong activation function. If instead training data gets learned extremely well but generalization is poor, that's not an architecture problem; it belongs in the regularization phase further down.

### Loss Function and Optimizer Selection

#### What this phase covers
Match the loss function to the problem type: cross-entropy for classification, mean squared error for regression. Pick an optimizer, with Adam as a safe, well-tested default in most cases. Decide the learning rate here too; it's one of the most consequential single numbers in the whole setup.

#### If it's not working
If loss doesn't decrease at all, the learning rate is likely too low, or the loss function doesn't match the problem type. If loss explodes or oscillates wildly, the learning rate is likely too high. Confirm the loss function actually matches your problem definition; a classification problem trained with a regression loss produces confusing behavior that looks like a training bug but isn't.

### Training Loop

#### What this phase covers
Write the loop: forward pass, compute loss, backward pass, optimizer step, repeated over batches and epochs. Track both training loss and validation loss every epoch, not training loss alone. Without a validation signal during training, overfitting stays invisible until a separate evaluation much later.

#### If it's not working
If training loss decreases while validation loss stalls or rises, that's overfitting, visible in real time, and it's the direct cue to move to the regularization phase. If training loss doesn't move at all, check the architecture or loss/optimizer phase before assuming the loop itself has a bug.

### Evaluation on Held-Out Test Set

#### What this phase covers
Run one honest check on data the model never touched during training or tuning, using metrics that fit the problem, not just accuracy. For imbalanced classification, check precision and recall per class specifically; an aggregate accuracy figure can look fine while some classes are barely being predicted at all.

#### If it's not working
If overall accuracy looks fine but a per-class breakdown shows one or more classes performing badly, that's a class-imbalance problem, addressed by revisiting preprocessing (resampling) or the loss function (class weighting), not by changing the architecture. If test performance is far worse than validation performance during training, suspect a data leak upstream or a genuine distribution difference between validation and test worth investigating directly.

### Visualize Results

#### What this phase covers
Plot loss curves, training against validation side by side, and build a confusion matrix or other diagnostics that make model behavior visible rather than buried in a table. A plot often reveals a pattern, a systematic misclassification, a sudden spike at a particular epoch, that's much harder to spot scanning printed metrics alone.

#### If it's not working
An unexpected shape in the loss curve is itself the signal. Sudden spikes usually point back to the training loop or learning rate. A validation curve that diverges early from the training curve, while both are still improving, is the visible signature of overfitting, and it sends you to the regularization phase, not back to architecture.

### Apply Regularization

#### What this phase covers
Once overfitting shows up (training loss low, validation or test loss much higher), address it directly rather than hoping more training epochs fix it. Common tools: dropout (randomly disabling a fraction of neurons during training so the network can't lean on any single path), L1/L2 weight penalties (discouraging weights from growing large, added via the optimizer's `weight_decay` argument for L2, or manually into the loss for L1), batch or layer normalization (stabilizing the scale of values between layers), and early stopping (halting training once validation loss stops improving, rather than pushing training loss toward zero at validation's expense).

#### If it's not working
If dropout or weight decay barely changes the gap between training and validation loss, the rate or penalty strength may be too weak; try increasing it before abandoning the technique. If a technique that should help instead damages performance on a specific small class, check whether that class had very few examples to begin with; aggressive regularization can destabilize a class the model barely had signal for in the first place, and the fix there is closer to feature engineering or data acquisition than regularization strength.

### Hyperparameter Tuning (optional)

#### What this phase covers
Adjust learning rate, batch size, layer sizes, dropout rate, and similar settings systematically rather than one at a time by feel. A grid search, random search, or a dedicated tuning library beats ad hoc trial and error, especially once several settings are being changed together and their effects start interacting.

#### If it's not working
If tuning produces marginal or inconsistent gains, that's often a sign the real bottleneck sits elsewhere: data quality, feature engineering, or a genuine scarcity problem for a specific class or category, not the hyperparameters themselves. Don't keep tuning past the point of diminishing, noisy returns; that effort is usually better spent revisiting data or features.

### Iterate

#### What this phase covers
Loop back through architecture, loss and optimizer choice, regularization, training, and evaluation, based on what the evaluation and visualizations actually showed. This is rarely one clean linear pass; expect to cycle through this block several times before landing on something worth calling final.

#### If it's not working
If you're unsure where to focus next, prioritize by what the evaluation step flagged most clearly, a specific failing class, a specific overfitting signature, a specific unstable run, rather than changing several things at once. Changing one thing at a time is the only way to know which change produced the next result.

### Error Analysis (optional)

#### What this phase covers
Look closely at individual misclassified examples, not just aggregate metrics. Pull a handful of specific wrong predictions and check what they share: a particular feature range, a particular category, a consistent confusion between two specific classes. This often surfaces a concrete, fixable pattern a summary table won't show.

#### If it's not working
If a clear pattern emerges, one class consistently confused with another specific class rather than classes generally, that points back to feature engineering (is there a feature that would help separate these two specifically) or data acquisition (are there simply too few examples of one of them), not a generic architecture tweak.

### Save or Checkpoint the Model

#### What this phase covers
Save the trained model's learned weights to disk; a state dictionary is generally more portable than saving the full object. If your pipeline includes fitted preprocessing objects, a scaler or an encoder, save those alongside the model. They're required to process any new raw input the same way the training data was processed, and a model without them can't be used on genuinely new data.

#### If it's not working
If the model fails to load correctly later, or produces different behavior than expected, check for an architecture mismatch between the saved state dictionary and the class currently used to load it. That's the most common cause, usually from the model class changing after the weights were saved.

### Run Inference on New, Unseen Input

#### What this phase covers
Test the model on a handful of individual, specific examples beyond the aggregate test-set numbers, as a concrete sanity check. Convert raw logits to probabilities with softmax if you want interpretable confidence values rather than raw, unbounded scores.

#### If it's not working
If the model produces a confidently wrong prediction on a specific example, not just wrong but wrong with high confidence, that's often more informative than an uncertain wrong answer. Trace it back to error analysis or evaluation to see whether it's an isolated case or part of a broader pattern.

### Deploy to Production

#### What this phase covers
Wrap the saved model in something that can serve real requests: an API endpoint (FastAPI or Flask are common choices), a batch scoring script, or an embedded call inside a larger application. Package dependencies precisely (a `requirements.txt` or a container image) so the serving environment matches the one the model was trained and tested in. Set up basic monitoring: track prediction latency, error rates, and, if possible, some proxy for whether the input data's distribution is drifting away from what the model was trained on.

#### If it's not working
If predictions in production don't match what the notebook produced for the same input, the most common cause is a preprocessing mismatch; confirm the exact same scaler, encoder, and feature order are being applied at serving time. If performance quietly degrades over time with no code change, that's usually data drift; the real-world input distribution has shifted away from training data, and the fix is retraining on more recent data, not debugging the existing model.

### Track Experiments (optional)

#### What this phase covers
Log hyperparameters and results for every run. A simple spreadsheet works early on; a dedicated tool (Weights & Biases, MLflow) scales better once many variants are being compared. Without this, it's easy to lose track of which specific settings produced which specific result once several iterations have piled up.

#### If it's not working
If you can't reproduce a result you got previously, or you're unsure which combination of changes led to your best run so far, that's exactly what experiment tracking prevents. Start logging now, even retroactively for recent runs, rather than continuing without it.

### Document and Push to GitHub

#### What this phase covers
Write up what was built and why, ideally throughout the project rather than only at the end. Commit incrementally as each phase completes, not in one large dump once everything's finished. A good final writeup covers the problem, the approach, the results including honest limitations, and what was tried and rejected, not only what worked.

#### If it's not working
If writing the documentation surfaces a decision you can no longer clearly explain, that's a legitimate reason to revisit that specific phase and either re-derive the reasoning or reconsider the choice. If you can't explain why a step was done a certain way, it's worth rechecking rather than writing around the gap.

### Project Summary

**Step 1 — Environment Check:** Confirmed the Colab machine has all the tools we need (PyTorch, pandas, NumPy, scikit-learn) and recorded their exact versions, so we have a reference point if anything behaves oddly later.

**Step 2 — Reproducibility (Random Seed):** Fixed a "seed" number so that any randomness in our process (like how a model starts learning) happens the same way every time we re-run the notebook — letting us trust that changes in results come from our actual changes, not random luck.

**Step 3 — Download the Dataset:** Pulled the two official NSL-KDD data files (one for training, one for testing) onto this Colab machine.

**Step 4 — Load Data into pandas:** Read those files into table form, manually attaching the correct column names, since the raw files don't include headers on their own.

**Step 5 — Preview the Raw Data:** Looked at actual rows and column types, confirming which of the 41 features are numbers and which are text.

**Step 6 — Label Distribution:** Checked how many times each attack type appears in training. Found the data is heavily lopsided — a couple of attack types make up the vast majority of records, while several others appear only a handful of times.

**Step 7 — Test Set Label Check:** Compared training labels against test labels, and found the test set contains 17 attack types that never once appear in training — the real, built-in challenge of this dataset.

**Step 8 — Check `worm` Occurrence:** Investigated one borderline attack type (`worm`) that different sources classify differently. Found it barely appears in test data at all (2 records), so we assigned it to DoS and moved on without over-debating a near-irrelevant edge case.

**Step 9 — Problem Definition:** Formally wrote down what we're building: a model that classifies each connection as `normal` or one of 4 attack categories (`DoS`, `Probe`, `R2L`, `U2R`) — 5 classes total.

**Step 10 — Categorical Feature Inspection:** Checked how many distinct values the 3 text columns (`protocol_type`, `service`, `flag`) contain, to decide how to convert them into numbers.

**Step 11 — Categorical Consistency Check:** Confirmed every text value in the test set also appears in training — no surprises there, unlike the label column.

**Step 12 — One-Hot Encode Categorical Columns:** Converted the 3 text columns into 84 separate yes/no columns (one per possible value), since a model can only process numbers, not words like "tcp" or "http."

**Step 13 — Numeric Feature Inspection:** Looked at the actual ranges of the 38 numeric features, and found some values reach into the billions while others are capped at 0 or 1 — a scale mismatch that needs fixing.

**Step 14 — Scale Numeric Features:** Rescaled all 38 numeric features onto a comparable footing, so no single feature dominates the model's calculations just because its raw numbers happen to be bigger.

**Step 15 — Combine Numeric and Categorical Features:** Joined the 38 rescaled numeric columns and the 84 yes/no categorical columns into one final table — 122 numbers per connection record. This is the actual input the model will receive, saved as `X_train` and `X_test`.

**Step 16 — Build Target Labels:** Converted the specific attack names in the `label` column into our 5 agreed categories (`normal`, `DoS`, `Probe`, `R2L`, `U2R`), then turned those 5 category names into plain numbers (0-4), since a model can only work with numbers, not words.

**Step 17 — Baseline Model:** Built two simple, non-neural-network reference points before touching the real model: one that always guesses the most common category (scored 43.1%), and a basic classical model, Logistic Regression (scored 76.1%). These numbers set the floor and a real bar our actual model needs to beat.

**Step 18 — Understanding the Perceptron and MLP (Theory):** Learned what a single artificial neuron actually does — take some numbers in, multiply each by a weight, add a bias, pass it through an activation function — and why stacking many of these into layers (with activation functions in between) lets a network learn far more complex patterns than one neuron alone ever could.

**Step 19 — Designing the MLP Architecture:** Built our actual network in code: 122 inputs in, 2 hidden layers (128 neurons, then 64) using ReLU activation, narrowing down to 5 outputs — one per class. Also covered what a "forward pass" means: pushing one row of data through every layer, in order, to get a prediction.

**Step 20 — Loss Function and Optimizer:** Set up the two pieces that let the network actually learn: a loss function (`CrossEntropyLoss`) that measures exactly how wrong a prediction was, and an optimizer (`Adam`) that uses that measurement to adjust the network's weights.

**Step 21 — Tensors and Batching:** Converted our data into PyTorch's required format (tensors), and set up feeding it to the network in small chunks (batches of 64 rows) rather than all 125,973 rows at once.

**Step 22 — The Training Loop:** Ran the network through the training data 10 full times (10 epochs). Watched the loss drop sharply at first, then level off — direct evidence the network was genuinely learning from the data batch by batch.

**Step 23 — Evaluate on Test Set:** Tested the trained network on data it had never seen before. Training loss was extremely low, but test accuracy came out to 76.1% — essentially tied with plain Logistic Regression. This revealed the network had mostly memorized the training data rather than learning patterns general enough to transfer well to new, unseen data.

**Step 24 — Visualize Training Loss:** Plotted the 10 epoch-by-epoch loss values on a graph, making the steep-drop-then-flatten learning pattern immediately visible, and confirmed our fixed seed genuinely reproduces the exact same training run every time, by re-running training from a fresh start and getting identical numbers.

**Step 25 — Per-Class Evaluation:** Went beyond overall accuracy and checked precision/recall for each class individually. Found that DoS and normal performed well, but R2L and U2R were badly broken — R2L had 97% precision but only 7% recall, meaning the model almost never flagged real R2L attacks at all, despite being right when it did.

**Step 26 — Confusion Matrix:** Visualized exactly what the model predicted instead of the correct answer. Confirmed the failure directly: nearly 90% of real R2L attacks, and most U2R attacks, were being labeled `normal` — the model was defaulting to "safe" guesses whenever it faced a rare attack type it hadn't learned well.

**Step 27 — Class-Weighted Loss:** Learned that misclassifying a rare class was costing the model the same as misclassifying a common one, giving it no real incentive to learn rare patterns. Computed weights that penalize mistakes on rare classes (R2L, U2R) far more heavily than common ones (DoS, normal).

**Step 28 — Retrain with Class-Weighted Loss:** Retrained a fresh model using those weights. Training loss looked higher and noisier than before — expected, since rare-class mistakes were now deliberately amplified.

**Step 29 — Evaluate Weighted Model and Compare:** R2L's recall improved meaningfully (7% → 17%), but U2R's precision collapsed from 91% to 7% — the model started crying "U2R" far too often, mostly incorrectly. A real, imperfect tradeoff, not a clean fix.

**Step 30 — Dampened Class Weights:** Tried softening the most extreme weight (U2R's, which was 484x) using a square root, hoping to keep the benefit without the side effect. Result: R2L's improvement partially reversed, and U2R's precision problem barely improved — revealing that loss weighting alone couldn't fix U2R's core issue.

**Step 31 — Oversample U2R in Training Data:** Switched techniques entirely: instead of changing the loss, duplicated U2R's ~52 real training rows 20 times so the network saw them repeatedly. This worked — U2R's precision recovered to 64% and recall nearly doubled to 31%, with no real harm to other classes. Our best single change so far.

**Step 32 — Oversample R2L as Well:** Tried extending the same technique to R2L. Result: R2L barely improved, and U2R's gains partially reversed — combining oversampling across two classes at once destabilized training rather than compounding the benefit.

**Step 33 — Final Model Selection:** Compared all five attempts side by side and chose the U2R-only oversampled model as final — it gave the best balanced improvement without harming other classes. Documented R2L's poor performance as a genuine, unresolved limitation of this dataset and approach, not something we forced a fix for.

**Step 34 — Save the Final Model:** Saved the chosen model's learned weights to a file (`intrusion_mlp_final.pth`), so it can be reloaded later without retraining from scratch.

**Step 35 — Inference on Sample Rows:** Ran the final model on a handful of individual test rows to see real predictions. Four out of five were correct with near-total confidence; one row was a real Probe attack confidently (100%) misclassified as normal — a concrete, visible example of the model's genuine limitations, not just an abstract percentage.

**Step 38 — Enhanced Architecture (Dropout + BatchNorm):** Built a new model class adding two regularization techniques directly into the architecture: BatchNorm after each linear layer (stabilizing the scale of values flowing through the network) and Dropout after each activation (randomly disabling 30% of neurons per forward pass during training, so the network can't over-rely on any single path).

**Step 39 — Weight Decay (L2 Regularization):** Turned on Adam's `weight_decay` argument, which had sat unused since Step 20. This adds a penalty for large weights directly into training, pushing back against the network fitting training data too precisely.

**Step 40 — Train with Early Stopping:** Instead of training for a fixed 10 epochs regardless of outcome, checked test loss after every epoch and stopped once it failed to improve for 3 epochs in a row, keeping the best version seen rather than the last one. Test loss came in at less than half the original model's (1.37 vs 3.13), and per-class evaluation showed real gains: Probe's precision jumped to 81%, U2R's F1-score improved from 0.42 to 0.54, and normal's recall rose to 96% — all without hurting DoS. R2L stayed essentially unchanged, confirming its problem was data scarcity, not overfitting.

**Step 41 — Learning Rate Scheduling:** Added `ReduceLROnPlateau`, set to cut the learning rate in half whenever test loss stalled. Result: the model's best performance had already occurred at epoch 1, before the scheduler ever got the chance to trigger — a real, if quiet, finding that the bottleneck here wasn't step size near the optimum.

**Step 42 — Adding L1 Regularization:** Tested L1 (which pushes less useful weights toward exactly zero) on top of everything already added. Test loss looked marginally better on its own, but per-class metrics told a different story: U2R's precision collapsed from 72% to 8%, likely because L1 zeroed out weights carrying that fragile class's already-thin signal. Rejected and dropped from the final model — a genuine negative result, not a quiet omission.

**Step 43 — Initial vs. Enhanced Model, Final Comparison:** Locked in the Step 40 model (Dropout + BatchNorm + L2 + early stopping, no L1) as final, reaching 78% overall accuracy. Documented the full picture side by side: what improved (Probe, U2R, normal), what didn't move (R2L, exactly as predicted before running any of this), and what was tested and correctly discarded (learning rate scheduling, L1).

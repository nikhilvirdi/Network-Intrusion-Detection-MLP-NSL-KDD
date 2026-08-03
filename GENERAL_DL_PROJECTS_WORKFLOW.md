## Deep Learning Project Workflow

A reusable reference for building deep learning projects, structured as the order these steps actually happen in.

1. **Set up version control.** Create the GitHub repo early, even empty, so every stage of the project is tracked from the start.
2. **Set up the environment.** Colab or a local notebook, imports, and a GPU check if you need one.
3. *(optional)* **Set random seeds.** Fix the seed for NumPy, PyTorch, and any split functions so results are reproducible across runs, since neural net weight initialization and data shuffling both involve randomness.
4. **Define the problem precisely.** Classification or regression, how many classes, what "success" looks like. This decides your loss function and evaluation metric before any model code is written.
5. **Acquire and explore the data (EDA).** Load the raw data and look at it: size, feature types, class balance, missing values, distributions. No model code until you've actually seen what you're working with.
6. **Feature engineering.** Encode categorical columns, derive or combine features where useful, and drop redundant or near-constant ones. This matters a lot for tabular data, since MLPs have no built-in structure to exploit rows-and-columns patterns. It matters far less for raw images or text, where the network learns its own features.
7. **Preprocess the data.** Scale or normalize numeric features, handle missing values, and split into train, validation, and test sets.
8. *(optional)* **Train a quick baseline model.** Logistic regression or a decision tree, before touching deep learning. This tells you whether DL is even worth the added complexity and gives you a reference number to beat.
9. **Design the model architecture.** Number of layers, size of each layer, activation functions, decided before writing the class.
10. **Choose the loss function and optimizer.** Matched to the problem type: cross-entropy for classification, MSE for regression, with Adam as the default optimizer.
11. **Write the training loop.** Forward pass, compute loss, backward pass, optimizer step, repeated over batches and epochs, tracking both training and validation loss every epoch.
12. **Evaluate on the held-out test set.** A separate, honest check using metrics that fit the problem, not just accuracy.
13. **Visualize results.** Loss curves, confusion matrix, or other diagnostics that make the model's behavior visible.
14. *(optional)* **Tune hyperparameters.** Systematically adjust learning rate, batch size, layer sizes, and so on, rather than changing things ad hoc.
15. **Iterate.** Steps 9 through 13 loop based on what evaluation shows. It's rarely one linear pass.
16. *(optional)* **Run error analysis.** Look closely at specific misclassified examples to understand failure patterns, beyond aggregate metrics.
17. **Save or checkpoint the trained model.** So it can be reloaded later without retraining from scratch.
18. **Run inference on new, unseen input.** A sanity check beyond test-set numbers, using an example the model has never seen.
19. *(optional)* **Track experiments.** Log hyperparameters and results per run, even in a simple spreadsheet or a tool like Weights & Biases, so different attempts are comparable.
20. **Document and push to GitHub.** Ideally incrementally throughout, not just at the end.

### Network Intrusion Detection — MLP

A multi-layer perceptron built in PyTorch to classify network connection records as normal or one of several attack categories, using the NSL-KDD dataset (125,973 training records, 22,544 test records, 41 features).

Built as a hands-on project to learn core deep learning fundamentals: the forward/backward pass, loss functions and optimizers (cross-entropy, Adam), and overfitting and regularization. NSL-KDD's test set is deliberately harder than its training set, so there's a real generalization gap to observe here rather than a hypothetical one.

#### Stack
- PyTorch
- Google Colab
- pandas / scikit-learn (preprocessing)
- matplotlib / seaborn (visualization)

#### Dataset
[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) is a refined, de-duplicated version of the original KDD Cup 99 dataset.

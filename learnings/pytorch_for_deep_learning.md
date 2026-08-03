## PyTorch for Deep Learning

PyTorch is where the actual network gets built, trained, and run — tensors,
automatic differentiation, layers, loss functions, optimizers, and the
training loop that ties them together. This file covers the core surface
that shows up in nearly every PyTorch project, general enough to apply to
any deep learning task, with runnable code and real output for each piece.

### Tensors: The Foundation

**Theory:** a tensor is PyTorch's core data structure — conceptually the
same as a NumPy array (a grid of numbers), but with two extra abilities
NumPy arrays don't have: automatic gradient tracking (needed for
backpropagation) and the ability to run on a GPU. Every input, weight, and
intermediate value in a PyTorch model is a tensor.

```python
import torch

t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(t)
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])
print(t.shape)   # torch.Size([2, 3])
print(t.dtype)   # torch.float32

zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
rand = torch.rand(2, 3)
print(rand.shape)   # torch.Size([2, 3])
```

`torch.float32` is the standard precision for neural network computation —
enough accuracy for training, at half the memory of `float64`. Explicitly
setting `dtype=torch.float32` when creating tensors from other sources
avoids precision mismatches, since some sources default to `float64`.

**NumPy interop** — since most data preprocessing happens in NumPy/pandas
before reaching PyTorch:

```python
import numpy as np

arr = np.array([1.0, 2.0, 3.0])
t_from_np = torch.from_numpy(arr)
print(t_from_np, t_from_np.dtype)   # tensor([1., 2., 3.], dtype=torch.float64)

back_to_np = t_from_np.numpy()
print(back_to_np, type(back_to_np))   # [1. 2. 3.] <class 'numpy.ndarray'>
```

**Gotcha:** `torch.from_numpy` preserves the NumPy array's original dtype —
here `float64`, NumPy's own default — not PyTorch's `float32` default. If
you build tensors this way instead of `torch.tensor(arr, dtype=torch.float32)`,
you can end up with a `float64` input meeting `float32` model weights,
which raises a dtype-mismatch error the moment they interact. Explicitly
cast when converting from NumPy: `torch.from_numpy(arr).float()`, or specify
`dtype=torch.float32` when using `torch.tensor(...)` directly.

**Reshaping and indexing:**

```python
x = torch.arange(12)
print(x)   # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

reshaped = x.reshape(3, 4)
print(reshaped)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])
print(reshaped[0])     # tensor([0, 1, 2, 3])       — row 0
print(reshaped[:, 1])  # tensor([1, 5, 9])          — column 1

# x.reshape(3, 5)
# RuntimeError: shape '[3, 5]' is invalid for input of size 12
```
Reshape only works when the new shape's total element count matches the
original — `3 × 4 = 12` works, `3 × 5 = 15` doesn't, and PyTorch raises
immediately rather than silently truncating or padding.

**Basic operations:**

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([10.0, 20.0, 30.0])
print(a + b)             # tensor([11., 22., 33.])
print(a * b)              # tensor([10., 40., 90.])   — elementwise, not matrix multiply
print(torch.dot(a, b))    # tensor(140.)

m1 = torch.rand(2, 3)
m2 = torch.rand(3, 4)
print((m1 @ m2).shape)    # torch.Size([2, 4])         — matrix multiplication
```
`*` is elementwise multiplication; `@` (or `torch.matmul`) is matrix
multiplication — mixing these up is a common source of silent shape bugs,
since both often run without error but produce mathematically different,
wrong results.

### Autograd: Automatic Differentiation

**Theory:** `requires_grad=True` tells PyTorch to track every operation
performed on a tensor, building a computation graph behind the scenes.
Calling `.backward()` walks that graph in reverse, computing the gradient
(rate of change) of the final value with respect to every tracked tensor —
this is what makes backpropagation automatic rather than something you
derive and code by hand for every model.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)   # tensor(7.)   — dy/dx = 2x + 3, at x=2: 2(2)+3 = 7
```

**Turning tracking off, when it's not needed:**

```python
with torch.no_grad():
    z = x * 2
print(z.requires_grad)   # False

detached = (x * 2).detach()
print(detached.requires_grad)   # False
```
- `torch.no_grad()` — a context manager disabling gradient tracking for
  everything inside it. Used during evaluation/inference, since no
  `.backward()` call is coming, so tracking would only waste memory and
  computation.
- `.detach()` — pulls a single tensor out of the computation graph, useful
  when you want a value without letting gradients flow through that
  specific point.

### Building a Model: `nn.Module`

**Theory:** every PyTorch model is a class inheriting from `nn.Module` — not
boilerplate, but what gives the model automatic tracking of all its
learnable parameters, and enables the automatic backward pass to actually
work. `__init__` declares what layers the model *has*; `forward` declares
what happens to input data, in order.

```python
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = TinyMLP(10, 16, 3)
print(model)
# TinyMLP(
#   (fc1): Linear(in_features=10, out_features=16, bias=True)
#   (relu): ReLU()
#   (fc2): Linear(in_features=16, out_features=3, bias=True)
# )
print(sum(p.numel() for p in model.parameters()))   # 227

sample_input = torch.rand(5, 10)
out = model(sample_input)
print(out.shape)   # torch.Size([5, 3])
```

- `super().__init__()` — mandatory first line in any `nn.Module` subclass;
  skipping it silently breaks the model's ability to track its own
  parameters, a common and confusing beginner bug.
- `nn.Linear(in_features, out_features)` — one fully-connected layer:
  every input connects to every output neuron. Creates a weight matrix and
  bias vector, both auto-initialized with small random values.
- Calling `model(sample_input)` directly (not `model.forward(sample_input)`)
  is the correct way to run data through — it triggers some internal
  bookkeeping around `forward()` that calling `.forward()` directly skips.

**`nn.Sequential`** — a shorthand for simple, strictly linear stacks of
layers, when a full custom class with branching logic isn't needed:

```python
seq_model = nn.Sequential(
    nn.Linear(10, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)
print(seq_model(sample_input).shape)   # torch.Size([5, 3])
```

### Activation Functions

**Theory:** without a nonlinear activation between layers, stacking multiple
linear layers mathematically collapses into being equivalent to one single
layer — nonlinearity is what lets a multi-layer network learn curved,
complex relationships instead of only straight-line ones.

```python
import torch.nn.functional as F

sample = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
print(F.relu(sample))
# tensor([0.0000, 0.0000, 0.0000, 0.5000, 2.0000])
print(torch.sigmoid(sample))
# tensor([0.1192, 0.3775, 0.5000, 0.6225, 0.8808])
print(torch.tanh(sample))
# tensor([-0.9640, -0.4621,  0.0000,  0.4621,  0.9640])
print(F.gelu(sample))
# tensor([-0.0455, -0.1543,  0.0000,  0.3457,  1.9545])
```

- **ReLU** (`max(0, x)`) — the modern default for hidden layers; cheap to
  compute, and avoids a training problem (vanishing gradients) that older
  activations suffer from more severely.
- **Sigmoid** — squashes to (0, 1); used for binary classification output,
  rarely for hidden layers anymore since it saturates (flattens toward 0 or
  1) and can slow learning.
- **Tanh** — squashes to (-1, 1); zero-centered, an improvement over
  sigmoid for hidden layers historically, still less common than ReLU today.
- **GELU** — a smoother variant used heavily in transformer architectures;
  behaves similarly to ReLU for large inputs but transitions more smoothly
  near zero.

### Regularization Layers: `BatchNorm1d`, `Dropout`

**Theory:** these two layers combat overfitting — a model fitting training
data (including its noise) too precisely, at the cost of performing worse
on new data.

```python
class RegularizedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

reg_model = RegularizedMLP(10, 16, 3)
reg_model.train()
out_train = reg_model(sample_input)
reg_model.eval()
out_eval = reg_model(sample_input)
print(not torch.allclose(out_train, out_eval))   # True — behavior genuinely differs
```

- **`BatchNorm1d`** — normalizes a layer's output across the current batch
  toward zero mean, unit variance, before the activation. Stabilizes and
  generally speeds up training of deeper networks. `1d` specifically fits
  flat, tabular-style batches (batch, features) — `BatchNorm2d` is the
  image-shaped equivalent.
- **`Dropout(p)`** — randomly zeroes a fraction `p` of neurons on every
  forward pass during training, so the network can't over-rely on any
  single path. Automatically disabled during evaluation.
- **`model.train()` / `model.eval()`** — this is exactly why these two calls
  matter, in a way they didn't for a model without these layers: `Dropout`
  and `BatchNorm` behave differently depending on this mode. Forgetting to
  call `.eval()` before evaluation means dropout stays active and BatchNorm
  keeps using batch statistics instead of running averages — producing
  inconsistent, non-reproducible evaluation results.

### Loss Functions

**Theory:** a loss function converts "how wrong was this prediction" into a
single differentiable number that `.backward()` can use to compute
gradients. The right choice depends entirely on the problem type.

```python
criterion_ce = nn.CrossEntropyLoss()
logits = torch.tensor([[2.0, 0.5, 0.1], [0.2, 3.0, 0.1]])
targets = torch.tensor([0, 1])
print(criterion_ce(logits, targets).item())   # 0.21319009363651276

criterion_mse = nn.MSELoss()
preds_reg = torch.tensor([3.0, 5.0, 2.5])
targets_reg = torch.tensor([2.5, 5.0, 3.0])
print(criterion_mse(preds_reg, targets_reg).item())   # 0.1666666716337204

criterion_bce = nn.BCEWithLogitsLoss()
logits_binary = torch.tensor([2.0, -1.0, 0.5])
targets_binary = torch.tensor([1.0, 0.0, 1.0])
print(criterion_bce(logits_binary, targets_binary).item())   # 0.3047555983066559
```

- **`CrossEntropyLoss`** — multi-class classification. Takes raw logits
  directly (not probabilities) and internally applies softmax as part of
  its calculation — never add your own softmax before this loss, it would
  double up and produce incorrect gradients.
- **`MSELoss`** — regression (predicting a continuous number). Average
  squared difference between prediction and true value.
- **`BCEWithLogitsLoss`** — binary classification. Like `CrossEntropyLoss`,
  takes raw logits and applies sigmoid internally — don't apply sigmoid
  yourself beforehand.

**Converting logits to readable probabilities (for inference display, not
training):**

```python
probs = F.softmax(logits, dim=1)
print(probs)
# tensor([[0.7285, 0.1625, 0.1090],
#         [0.0545, 0.8962, 0.0493]])
print(probs.sum(dim=1))   # tensor([1., 1.])
```
`dim=1` applies softmax across each row's class values independently — not
across rows. `CrossEntropyLoss` already does this internally during
training; this manual call is purely for turning a trained model's output
into human-readable confidence values afterward.

**Gotcha — wrong target dtype:**

```python
bad_targets = torch.tensor([0.0, 1.0])   # float, wrong dtype for class labels
# criterion_ce(logits, bad_targets)
# RuntimeError: expected target dtype to be Long or Byte, but got Float
```
`CrossEntropyLoss` expects integer class indices (`torch.long`), not floats
— convert labels with `torch.tensor(y, dtype=torch.long)`.

### Optimizers

**Theory:** the optimizer uses gradients (computed by `.backward()`) to
actually update the model's weights, nudging them toward reducing the loss.

```python
import torch.optim as optim

model2 = TinyMLP(10, 16, 3)
opt_sgd = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
opt_adam = optim.Adam(model2.parameters(), lr=0.001, weight_decay=1e-4)
opt_adamw = optim.AdamW(model2.parameters(), lr=0.001, weight_decay=1e-2)

print(opt_adam.param_groups[0]['weight_decay'])   # 0.0001
```

- **`model.parameters()`** — inherited from `nn.Module`; returns every
  learnable weight and bias, telling the optimizer exactly what it's
  allowed to adjust.
- **`SGD` with `momentum`** — the classic optimizer; `momentum` accumulates
  a moving average of past gradients so updates don't zig-zag through
  narrow valleys in the loss landscape.
- **`Adam`** — combines momentum with per-parameter adaptive learning
  rates; the default choice in most modern deep learning, since it
  generally converges faster and more reliably than plain SGD with less
  manual tuning.
- **`weight_decay`** — L2 regularization, applied directly through the
  optimizer. Penalizes large weights, discouraging the network from fitting
  training data too precisely. `AdamW` implements weight decay slightly
  differently (decoupled from the gradient update) than plain `Adam` — the
  more theoretically correct version for weight decay specifically, and
  the more common default in newer projects.
- **`lr` (learning rate)** — controls how large each weight update step is.
  Too high causes unstable, oscillating training; too low causes painfully
  slow convergence. `0.001` is a common Adam starting point.

**Learning rate scheduling** — changing the learning rate during training,
rather than keeping it fixed the whole time:

```python
scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(opt_adam, mode='min', factor=0.5, patience=1)
scheduler_step = optim.lr_scheduler.StepLR(opt_adam, step_size=5, gamma=0.1)

print(opt_adam.param_groups[0]['lr'])   # 0.001
# ... inside training loop, after optimizer.step(): ...
# scheduler_step.step()
# after 6 epochs with step_size=5, gamma=0.1:
print(opt_adam.param_groups[0]['lr'])   # 0.0001  (cut by 10x once, after epoch 5)
```

- **`ReduceLROnPlateau`** — watches a metric (usually validation loss) and
  cuts the learning rate by `factor` whenever it stops improving for
  `patience` epochs. Call `.step(metric_value)` with the metric each epoch.
- **`StepLR`** — cuts the learning rate by `gamma` every `step_size` epochs,
  on a fixed schedule regardless of performance. Call `.step()` once per
  epoch, with no argument.
- **Order matters:** always call `optimizer.step()` before `scheduler.step()`
  within an epoch — calling them in the reverse order causes PyTorch to
  skip the first scheduled learning rate value, with a warning rather than
  a hard error, making it an easy mistake to miss.

### Data Handling: `Dataset`, `TensorDataset`, `DataLoader`

**Theory:** feeding an entire dataset through a network in one pass isn't
practical for most real data — `DataLoader` splits training data into
smaller batches, optionally shuffling their order each epoch.

```python
from torch.utils.data import TensorDataset, DataLoader

X_dummy = torch.rand(20, 10)
y_dummy = torch.randint(0, 3, (20,))
dataset = TensorDataset(X_dummy, y_dummy)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print(len(loader))   # 5   — 20 rows / batch_size 4
first_batch_X, first_batch_y = next(iter(loader))
print(first_batch_X.shape, first_batch_y.shape)   # torch.Size([4, 10]) torch.Size([4])
```

- **`TensorDataset(X, y)`** — pairs input rows with their labels so they
  stay correctly matched together through shuffling and batching.
- **`DataLoader(..., batch_size=4, shuffle=True)`** — `shuffle=True`
  randomizes row order every epoch, preventing the model from picking up
  on incidental ordering patterns rather than genuine feature relationships.
  Set `shuffle=False` for validation/test loaders, where order doesn't need
  randomizing and consistency can help with debugging.

### The Training Loop

**Theory:** every PyTorch training loop follows the same five-step cycle,
repeated for every batch, every epoch: zero gradients, forward pass,
compute loss, backward pass, optimizer step.

```python
model3 = TinyMLP(10, 16, 3)
optimizer3 = optim.Adam(model3.parameters(), lr=0.01)
criterion3 = nn.CrossEntropyLoss()

for epoch in range(3):
    total_loss = 0.0
    for batch_X, batch_y in loader:
        optimizer3.zero_grad()
        outputs = model3(batch_X)
        loss = criterion3(outputs, batch_y)
        loss.backward()
        optimizer3.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"epoch {epoch+1} avg loss: {avg_loss:.4f}")
# epoch 1 avg loss: 0.9856
# epoch 2 avg loss: 0.9048
# epoch 3 avg loss: 0.8544
```

- **`optimizer.zero_grad()`** — PyTorch accumulates gradients by default
  rather than replacing them; skipping this line means each batch's
  gradients incorrectly add onto the previous batch's, corrupting training.
  Must run before every batch's `.backward()` call.
- **`loss.backward()`** — triggers backpropagation across the whole
  network automatically, computing the gradient of the loss with respect
  to every parameter.
- **`optimizer.step()`** — actually updates every weight, using those
  gradients and the optimizer's own logic (momentum, adaptive rates, etc.).
- **`loss.item()`** — extracts a plain Python float from a single-value
  tensor, since `loss` itself is a tensor object.

**Evaluation, afterward:**

```python
model3.eval()
with torch.no_grad():
    test_out = model3(X_dummy)
    _, preds = torch.max(test_out, dim=1)
print(preds[:5])   # tensor([2, 2, 2, 2, 2])
```
`torch.max(tensor, dim=1)` returns both the max value and its index per
row; the index is the predicted class. `model.eval()` and `torch.no_grad()`
together are the standard pairing for any inference or evaluation code.

### Saving and Loading: `state_dict`

**Theory:** save a model's learned weights, not the full Python object —
more portable and stable across PyTorch versions.

```python
torch.save(model3.state_dict(), "tiny_mlp.pth")

model4 = TinyMLP(10, 16, 3)
model4.load_state_dict(torch.load("tiny_mlp.pth"))
# <All keys matched successfully>

model4.eval()
with torch.no_grad():
    out4 = model4(X_dummy)
print(torch.allclose(test_out, out4))   # True — identical weights, identical output
```

**Gotcha:** loading a `state_dict` into a model class whose architecture has
since changed (different layer sizes, renamed layers) raises a key-mismatch
error, or in rarer cases loads silently into the wrong shapes. Keep the
model class definition available alongside any saved weights file — the
weights alone don't describe the architecture they belong to.

**If your pipeline includes fitted preprocessing (scikit-learn scalers,
encoders):** save those alongside the model too. They're required to
process any new raw input identically to how training data was processed,
and a saved model without them can't correctly handle genuinely new,
unprocessed data.

### Device Handling (CPU / GPU)

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)   # cpu   (or cuda, if a GPU is available and detected)

model3.to(device)
X_dummy_dev = X_dummy.to(device)
print(X_dummy_dev.device)   # cpu
```
Both the model and every tensor fed into it must be on the same device —
mixing a CPU model with a GPU tensor (or vice versa) raises an error
immediately. Checking `torch.cuda.is_available()` rather than assuming a
GPU exists keeps the same code working correctly on either kind of machine.

### Common Mistakes and Debugging Tips

**Forgetting `optimizer.zero_grad()`:**

```python
model5 = TinyMLP(10, 16, 3)
opt5 = optim.SGD(model5.parameters(), lr=0.1)
crit5 = nn.CrossEntropyLoss()

out5a = model5(X_dummy)
crit5(out5a, y_dummy).backward()
grad_after_first = model5.fc1.weight.grad.clone()

out5b = model5(X_dummy)
crit5(out5b, y_dummy).backward()   # no zero_grad() called between these two
grad_after_second = model5.fc1.weight.grad.clone()

print(not torch.allclose(grad_after_first, grad_after_second))   # True — grads changed
print(torch.allclose(grad_after_second, grad_after_first * 2, atol=1e-4))   # True — accumulated
```
Without `zero_grad()`, gradients from consecutive `.backward()` calls add
together rather than replace each other — here, the second gradient is
almost exactly double the first, direct proof of silent accumulation. This
corrupts training gradually and doesn't throw any error, making it one of
the hardest PyTorch bugs to notice by symptom alone.

**Wrong target dtype for `CrossEntropyLoss`:** covered above — targets must
be `torch.long`, not float.

**Shape mismatches between layers:** `nn.Linear(128, 64)` following
`nn.Linear(122, 128)` works because `128` (output of the first) matches
`128` (input of the second) exactly. A typo or leftover value from a
previous architecture version here raises an immediate, clear shape error
at the mismatched layer — read the error's stated shapes carefully, they
point directly at which layer is wrong.

**Forgetting `.eval()` before evaluation:** covered above under
BatchNorm/Dropout — produces inconsistent results between runs, since
dropout keeps randomly zeroing neurons and BatchNorm keeps using
current-batch statistics instead of the running averages accumulated
during training.

**Low training loss, much higher validation/test loss:** not a PyTorch
bug at all — this is overfitting, and the fix is architectural/procedural
(dropout, weight decay, early stopping, more data), not something to debug
in the training loop's mechanics.

### Quick Recap

```text
torch.tensor(..., dtype=torch.float32)  — the core data structure; match
                                            dtype to avoid mismatches
requires_grad=True / .backward()        — automatic differentiation;
                                            torch.no_grad() to disable it
nn.Module + __init__ + forward          — how every model is defined;
                                            super().__init__() is mandatory
nn.Linear / ReLU / BatchNorm1d / Dropout — the common building blocks;
                                            train()/eval() changes their behavior
CrossEntropyLoss / MSELoss / BCEWithLogitsLoss — match the loss to the
                                            problem type; logits in, not
                                            probabilities
Adam / AdamW + weight_decay             — the standard optimizer choice;
                                            weight_decay is L2 regularization
TensorDataset + DataLoader(shuffle=True)   — batching and shuffling training data
zero_grad → forward → loss → backward → step — the five-step training
                                            loop cycle, every batch
model.eval() + torch.no_grad()          — the standard pairing for any
                                            evaluation or inference code
state_dict + torch.save/load            — the portable way to persist a
                                            trained model

Golden rule: optimizer.zero_grad() before every backward() call, and
model.eval() before every evaluation pass — the two easiest lines to
forget, and the two that most often cause silent, hard-to-trace bugs
rather than a clean crash.
```

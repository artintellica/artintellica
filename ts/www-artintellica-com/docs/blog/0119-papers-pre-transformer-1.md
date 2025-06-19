+++
title = "Influential Papers in Machine Learning – Pre-Transformer Era, Paper 1: A Fast Learning Algorithm for Deep Belief Nets (Hinton, Osindero & Teh, 2006)"
author = "Artintellica"
date = "2025-06-19"
+++

## Introduction

In 2006 Geoffrey Hinton, Simon Osindero, and Yee-Whye Teh published  
“A Fast Learning Algorithm for Deep Belief Nets.”  
The paper revived interest in _deep_ neural networks by proposing an
unsupervised, layer-by-layer pre-training procedure based on Restricted
Boltzmann Machines (RBMs).

Before this work, networks deeper than two or three layers were notoriously hard
to train with back-propagation alone—gradients vanished, local minima trapped
optimizers, and labeled data were scarce.  
Hinton et al. showed that a stack of RBMs could initialize deep networks in a
way that made subsequent supervised fine-tuning easy and effective.  
This single idea catalyzed the modern “deep learning” wave that eventually led
to AlexNet (2012), sequence-to-sequence models, and, ultimately, the
Transformer.

---

## ELI5 (Explain Like I’m 5)

Imagine you are building a very tall LEGO tower. If you try to build it from the
bottom to the top in one go, the pieces on top wobble and fall. A better
strategy is:

1. Build a solid base block.
2. Freeze it so it can’t break.
3. Build the next block on top of the frozen base.
4. Freeze that, and repeat.

Each _block_ is an RBM. After stacking several blocks, you _unfreeze_ the whole
tower and tweak the connections a little so everything fits perfectly.  
That’s exactly what Deep Belief Networks (DBNs) do for neural nets: they learn
one layer at a time (unsupervised), then tweak the whole thing (supervised).

---

## Mathematical Definitions

### Restricted Boltzmann Machine (RBM)

An RBM is an undirected graphical model with a layer of visible units
$v \in \{0,1\}^{D}$ and a layer of hidden units $h \in \{0,1\}^{F}$, with no
intra-layer connections.

Energy function:

$$
E(v,h) \;=\; -b^\top v \;-\; c^\top h \;-\; v^\top W h
$$

Probability distribution over $\{v,h\}$:

$$
P(v,h) \;=\; \frac{1}{Z}\; e^{-E(v,h)}
$$

where $Z = \sum_{v,h} e^{-E(v,h)}$ is the partition function.

Conditional independencies:

$$
P(h_j=1 \mid v) \;=\; \sigma\!\bigl(c_j + v^\top W_{\cdot j}\bigr),\quad
P(v_i=1 \mid h) \;=\; \sigma\!\bigl(b_i + W_{i\cdot}^\top h\bigr)
$$

with $\sigma(x)=1/(1+e^{-x})$.

### Contrastive Divergence (CD-$k$)

An approximate gradient for the log-likelihood:

$$
\frac{\partial \log P(v)}{\partial \theta} \;\approx\;
\mathbb{E}_{h\sim P(h\mid v)}\!\!\bigl[\tfrac{\partial E}{\partial \theta}\bigr]
\;-\;
\mathbb{E}_{v',h'\sim P_{k}(v',h')}\!\!\bigl[\tfrac{\partial E}{\partial \theta}\bigr]
$$

where $P_{k}$ runs $k$ steps of Gibbs sampling starting at the data $v$ (usually
$k=1$).

### Deep Belief Network (DBN)

A DBN of $L$ layers is a generative model:

$$
P(v,h^{(1)},\dots,h^{(L)}) \;=\;
P(v \mid h^{(1)}) \,
\Bigl[\prod_{l=1}^{L-2} P\bigl(h^{(l)} \mid h^{(l+1)}\bigr)\Bigr]\,
P\bigl(h^{(L-1)},h^{(L)}\bigr).
$$

The top two layers form an RBM; lower layers form directed, top-down generative
connections.

Greedy training algorithm:

1. Train first RBM on data $v$
2. Use its hidden probabilities as “data” to train second RBM, and so on.
3. Stack the learned weights to initialize a deep neural network.
4. Fine-tune using back-propagation.

---

## Further Mathematical Details

1. **Free Energy** for visible vector $v$:

   $$
   F(v) = -b^\top v - \sum_{j=1}^{F} \log\!\bigl(1 + e^{c_j + v^\top W_{\cdot j}}\bigr)
   $$

   $\;\;P(v) \propto e^{-F(v)}.$

2. **Why Greedy Works**  
   Hinton et al. prove that after training layer $l$, the variational lower
   bound on the log-likelihood of the data improves. Intuitively, each new RBM
   models the residual structure the previous layers could not capture.

3. **Fine-Tuning Objective**  
   If labels $y$ exist, append a softmax layer and minimize cross-entropy;
   otherwise, maximize joint log-likelihood (deep autoencoder).

---

## Further Conceptual Details

• **Unsupervised Pre-Training as Regularization**  
 The greedy layer-wise procedure initializes weights near a good basin of
attraction. With limited labeled data, this acts like a strong unsupervised
prior.  
• **Discriminative vs. Generative**  
 DBNs are generative, capable of _sampling_ new data, yet the same weights help
supervised tasks after fine-tuning.  
• **Historical Impact**  
 Practitioners adopted deep autoencoders, deep convolutional nets (with
unsupervised pre-training), and later deep supervised nets profiting from
smarter initialization (Xavier, He). Once better activation functions (ReLU),
data and compute scaled, the need for RBM pre-training faded—but the _depth_
revolution stayed.

---

## Code Demonstrations

Below we implement:

1. A binary RBM with CD-1 in PyTorch
2. A two-layer DBN (stacked RBMs)
3. Reconstruction of MNIST digits
4. Fine-tuning for classification

> Installation (recommendation):
>
> ```bash
> uv pip install torch torchvision matplotlib scikit-learn tqdm
> ```

```python
# rbm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

class RBM(nn.Module):
    def __init__(self, n_vis, n_hid, k=1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_vis, n_hid) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_vis))   # visible bias
        self.c = nn.Parameter(torch.zeros(n_hid))   # hidden bias
        self.k = k

    def sample_h(self, v):
        p = torch.sigmoid(F.linear(v, self.W.t(), self.c))
        return p, torch.bernoulli(p)

    def sample_v(self, h):
        p = torch.sigmoid(F.linear(h, self.W, self.b))
        return p, torch.bernoulli(p)

    def forward(self, v0):
        vk = v0.detach()
        for _ in range(self.k):
            hk_prob, hk = self.sample_h(vk)
            vk_prob, vk = self.sample_v(hk)
        return v0, vk_prob

    def free_energy(self, v):
        vbias_term = v @ self.b
        wx_b = F.linear(v, self.W.t(), self.c)
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)
        return -vbias_term - hidden_term

def train_rbm(rbm, train_loader, lr=1e-3, epochs=5):
    opt = torch.optim.Adam(rbm.parameters(), lr=lr)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (x, _) in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            x = x.view(x.size(0), -1)
            v0, vk = rbm(x)
            loss = torch.mean(rbm.free_energy(v0)) - torch.mean(rbm.free_energy(vk))
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f'  free-energy diff: {epoch_loss/len(train_loader):.4f}')
```

```python
# dbn_demo.py
import torch, matplotlib.pyplot as plt
from rbm import RBM, train_rbm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH = 128
transform = transforms.Compose([transforms.ToTensor(), lambda x: x > 0.5, lambda x: x.float()])
train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

rbm1 = RBM(28*28, 500, k=1)
train_rbm(rbm1, train_loader, epochs=5)

# Use hidden probs of rbm1 as data for rbm2
class HiddenDataset(torch.utils.data.Dataset):
    def __init__(self, base_loader, rbm):
        self.samples = []
        with torch.no_grad():
            for (x, _) in base_loader:
                x = x.view(x.size(0), -1)
                ph, _ = rbm.sample_h(x)
                self.samples.append(ph)
        self.data = torch.cat(self.samples)

    def __len__(self): return self.data.size(0)
    def __getitem__(self, idx): return self.data[idx], 0

hidden_ds = HiddenDataset(train_loader, rbm1)
hidden_loader = DataLoader(hidden_ds, batch_size=BATCH, shuffle=True)

rbm2 = RBM(500, 250, k=1)
train_rbm(rbm2, hidden_loader, epochs=5)

# Reconstruction test
def reconstruct(rbm1, rbm2, images, steps=1):
    v = images.view(images.size(0), -1)
    with torch.no_grad():
        ph1, h1 = rbm1.sample_h(v)
        ph2, h2 = rbm2.sample_h(ph1)
        # Top-down
        pv1, v1 = rbm2.sample_v(h2)
        pv0, v0 = rbm1.sample_v(pv1)
    return v, v0

test_ds = datasets.MNIST('.', train=False, download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=10, shuffle=True)
imgs, _ = next(iter(test_loader))
orig, recon = reconstruct(rbm1, rbm2, imgs)

fig, axes = plt.subplots(2, 10, figsize=(10,2))
for i in range(10):
    axes[0,i].imshow(orig[i].view(28,28), cmap='gray'); axes[0,i].axis('off')
    axes[1,i].imshow(recon[i].view(28,28), cmap='gray'); axes[1,i].axis('off')
plt.suptitle('Original (top) vs Reconstruction (bottom)')
plt.show()
```

Result: the bottom row digits look like blurred versions of the
originals—evidence that our DBN learned a generative model.

---

## Code Exercises

### Exercise 1 – Gaussian-Visible RBM

Modify the RBM so that visible units are real-valued with Gaussian noise,
suitable for pixel intensities in $[0,1]$.

Full solution:

```python
# gaussian_rbm.py
import torch, torch.nn as nn, torch.nn.functional as F

class GaussianRBM(nn.Module):
    def __init__(self, n_vis, n_hid, sigma=1.0, k=1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_vis, n_hid)*0.01)
        self.b = nn.Parameter(torch.zeros(n_vis))
        self.c = nn.Parameter(torch.zeros(n_hid))
        self.sigma2 = sigma**2
        self.k = k

    def sample_h(self, v):
        # v is real-valued
        p = torch.sigmoid((F.linear(v/self.sigma2, self.W.t(), self.c)))
        return p, torch.bernoulli(p)

    def sample_v(self, h):
        mean = F.linear(h, self.W, self.b)
        v = mean + torch.randn_like(mean) * self.sigma2**0.5
        return mean, v

    def forward(self, v0):
        vk = v0
        for _ in range(self.k):
            _, hk = self.sample_h(vk)
            _, vk = self.sample_v(hk)
        return v0, vk

    def free_energy(self, v):
        term1 = ((v - self.b)**2).sum(1)/(2*self.sigma2)
        term2 = torch.sum(F.softplus(F.linear(v/self.sigma2, self.W.t(), self.c)), 1)
        return term1 - term2
```

Usage: replace `RBM` with `GaussianRBM`; remove the binary threshold transform
on MNIST.

---

### Exercise 2 – Build a 3-Layer DBN and Fine-Tune for Classification

```python
# dbn_finetune.py
import torch, torch.nn as nn, torch.nn.functional as F
from rbm import RBM, train_rbm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH=128
transform = transforms.Compose([transforms.ToTensor(), lambda x: x>0.5, lambda x: x.float()])
train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

# Pre-training
rbm1 = RBM(784, 500); train_rbm(rbm1, train_loader, epochs=3)
hidden1 = torch.vstack([rbm1.sample_h(x.view(-1,784))[0] for x,_ in train_loader])

ds2 = torch.utils.data.TensorDataset(hidden1, torch.zeros(len(hidden1)))
loader2 = DataLoader(ds2, batch_size=BATCH, shuffle=True)

rbm2 = RBM(500, 250); train_rbm(rbm2, loader2, epochs=3)

hidden2 = torch.vstack([rbm2.sample_h(h)[0] for h in DataLoader(hidden1, batch_size=BATCH)])
ds3 = torch.utils.data.TensorDataset(hidden2, torch.zeros(len(hidden2)))
loader3 = DataLoader(ds3, batch_size=BATCH, shuffle=True)

rbm3 = RBM(250, 100); train_rbm(rbm3, loader3, epochs=3)

# Build a feed-forward NN initialized with RBM weights
class DBNClassifier(nn.Module):
    def __init__(self, rbm1, rbm2, rbm3):
        super().__init__()
        self.l1 = nn.Linear(784, 500)
        self.l2 = nn.Linear(500, 250)
        self.l3 = nn.Linear(250, 100)
        self.out = nn.Linear(100, 10)
        # copy weights
        self.l1.weight.data = rbm1.W.t()
        self.l1.bias.data = rbm1.c
        self.l2.weight.data = rbm2.W.t()
        self.l2.bias.data = rbm2.c
        self.l3.weight.data = rbm3.W.t()
        self.l3.bias.data = rbm3.c

    def forward(self,x):
        x = x.view(-1,784)
        x = torch.sigmoid(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return self.out(x)

model = DBNClassifier(rbm1, rbm2, rbm3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    total = correct = 0
    for (x,y) in DataLoader(train_ds, batch_size=BATCH, shuffle=True):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        preds = logits.argmax(1)
        total += y.size(0); correct += (preds==y).sum().item()
    print(f'Epoch {epoch+1}: train acc {correct/total:.4f}')
```

You should observe >95 % accuracy after a few epochs—even though we began with
_unsupervised_ pre-training.

---

### Exercise 3 – Visualize First-Layer Filters

```python
# visualize_filters.py
import matplotlib.pyplot as plt, math
from rbm import RBM, train_rbm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(), lambda x: x>0.5, lambda x: x.float()])
ds = datasets.MNIST('.', train=True, download=True, transform=transform)
loader = DataLoader(ds, batch_size=64, shuffle=True)

rbm = RBM(784, 100, k=1)
train_rbm(rbm, loader, epochs=5)

W = rbm.W.data.t().view(-1, 28,28)  # shape (100,28,28)
n = int(math.sqrt(100))
fig, axes = plt.subplots(n, n, figsize=(8,8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(W[i], cmap='gray'); ax.axis('off')
plt.suptitle('Learned 1st-Layer Features')
plt.show()
```

You’ll see stroke-like filters resembling pen curves—evidence of meaningful
feature discovery.

---

## Conclusion

Hinton, Osindero, and Teh’s 2006 paper re-introduced deep learning to the world
by solving the training bottleneck of very deep networks. The key ideas:

• Stack RBMs and train them greedily with Contrastive Divergence  
• Use unsupervised pre-training to initialize a deep net  
• Fine-tune with supervised back-propagation

Although modern practice rarely employs RBMs, the concepts of layer-wise
initialization, unsupervised representation learning, and energy-based models
still influence contemporary research (e.g., self-supervised learning, diffusion
models).  
Understanding this seminal work provides crucial historical context for the
evolution of deep neural networks—paving the way for everything that came after,
including the Transformer revolution.

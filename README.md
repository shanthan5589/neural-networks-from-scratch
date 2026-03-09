# Neural Networks from Scratch

Implementing neural networks using only NumPy — no PyTorch, no TensorFlow, no abstractions.

The goal of this repo is not to achieve state-of-the-art results. It is to understand exactly what happens inside a neural network — every matrix multiplication, every derivative, every weight update — by building it from the ground up.

---

## Why from Scratch?

Using a framework like PyTorch, you call `loss.backward()` and gradients appear. This repo is about understanding what `loss.backward()` actually does — deriving and implementing every gradient by hand using the chain rule.

---

## Progression

The three implementations follow a deliberate progression, each motivated by the failure of the previous one:

### 1. Logistic Regression
A single neuron. The simplest possible classifier.

Learns that a single linear decision boundary is not enough for real-world image data. Achieves 70% test accuracy on the Cat vs Non-Cat dataset — not because the implementation is wrong, but because the model itself is fundamentally limited.

### 2. Shallow Neural Network
One hidden layer with tanh activation, sigmoid output.

Adds non-linearity — the network can now learn curved decision boundaries. Understands why activation functions exist, why zero-initialization fails, and why tanh is preferred over sigmoid for hidden layers.

### 3. Deep Neural Network
L hidden layers, generalized with for loops.

Implements the full forward and backward pass for an arbitrary number of layers. Understands vanishing gradients, why deep networks need careful initialization (`* 0.01`), and why even a deep fully connected network cannot solve image classification properly without spatial awareness.

---

## What's Actually Implemented

Every line of the following is written from scratch:

- Sigmoid, tanh activation functions and their derivatives
- Binary cross-entropy loss
- Forward propagation for L layers
- Backpropagation — full chain rule derivation layer by layer
- Gradient descent parameter updates
- Weight initialization strategies

---

## Results

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | 99.04% | 70.00% |
| Shallow Neural Network | 94.26% | 62.00% |
| Deep Neural Network | 65.55% | 34.00% |
---

Dataset: Cat vs Non-Cat (209 training examples, 50 test examples, 64×64 RGB images)



## Key Concepts Understood

**Why dZ = A - Y at the output layer**
The chain rule applied to sigmoid + cross-entropy at the output layer simplifies to this. It is not a definition — it is a derived result specific to the output layer. For hidden layers, dZ is computed differently: `dZ[l] = W[l+1].T · dZ[l+1] * g'(Z[l])` — the error signal flows back from the next layer through its weights, scaled by the local activation derivative.

**Why hidden layers need tanh, not sigmoid**
Sigmoid outputs values between 0 and 1 — always positive. This causes zig-zag gradient updates. Tanh outputs between -1 and 1, is zero-centered, and converges faster.

**Why weights cannot be initialized to zero**
Zero initialization makes every neuron in a layer identical — they compute the same thing, receive the same gradients, and never differentiate. Random initialization breaks this symmetry.

**Why deep FC networks fail on images**
Flattening an image destroys spatial structure. More layers add representational power but cannot recover positional information that was lost. The solution is convolutional layers — covered in the next stage of this project.

**The vanishing gradient problem**
In deep networks, gradients are multiplied layer by layer during backprop. Sigmoid saturates near 0 and 1, producing near-zero derivatives. Multiplied across many layers, the gradient reaching early layers approaches zero — those layers stop learning.

---

## Repository Structure

```
neural-networks-from-scratch/
│
├── datasets/                        ← Cat vs Non-Cat dataset (h5 files)
│
├── 01_logistic_regression/
│   ├── logistic_regression.ipynb   ← implementation
│   └── README.md                   ← math, derivations, results
│
├── 02_shallow_neural_network/
│   ├── shallow_nn.ipynb
│   └── README.md
│
└── 03_deep_neural_network/
    ├── deep_nn.ipynb
    └── README.md
```

---

## Running the Notebooks

Clone the repo and launch Jupyter from the root folder:

```bash
git clone https://github.com/shanthan5589/neural-networks-from-scratch
cd neural-networks-from-scratch
jupyter notebook
```

The dataset paths are resolved automatically relative to the repo root.

---

## What's Next

These three models establish the foundation. The logical next step is Convolutional Neural Networks — which solve the spatial awareness problem that fully connected networks cannot.

That work is part of an ongoing mitosis detection project using histology images from the [CCMCT dataset](https://www.kaggle.com/datasets/marcaubreville/mitosis-wsi-ccmct-training-set/data).

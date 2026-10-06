# TP1 — Deep Learning with Keras, TensorFlow, and PyTorch

This folder contains the same beginner XOR neural-network exercise implemented in three frameworks:

- `keras_exercise.py`
- `tensorflow_exercise.py`
- `pytorch_exercise.py`

## Goal

Train a small neural network to learn XOR:

| x1 | x2 | y |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Architecture:

```
2 inputs -> 2 hidden neurons -> 1 output
```

The examples demonstrate:
- inputs and targets
- weights and biases
- hidden layer
- sigmoid activation
- forward propagation
- loss calculation
- backpropagation
- gradient descent / optimizer
- training epochs
- final predictions

## Install

```bash
pip install numpy tensorflow torch
```

## Run

```bash
python tp1/keras_exercise.py
python tp1/tensorflow_exercise.py
python tp1/pytorch_exercise.py
```

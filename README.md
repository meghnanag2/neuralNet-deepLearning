# EuroSAT Land Cover Classification — Manual CNN Optimization Study

This project implements an end-to-end image classification pipeline on the EuroSAT dataset (27,000 RGB satellite images, 10 classes) under strict manual training constraints.

The objective was to analyze how activation functions, batch size, momentum, and skip connections affect optimization behavior and gradient flow in deep networks.



## Dataset

- Source: torchvision.datasets.EuroSAT  
- Classes: 10 land-cover categories  
- Images: 64×64 RGB  
- Split: 70% train / 30% test (fixed seed)



## Key Experiments

- Shallow MLP (difficulty check)
- Baseline Deep CNN
- Activation comparison (Sigmoid vs Tanh vs LeakyReLU)
- Mini-batch SGD analysis
- Momentum tuning (α = 0.5, 0.9, 0.95)
- Extended deep model (+10 layers)
- Skip connection configurations
- Gradient magnitude analysis (epoch 1)



## Results Summary

![Results Table](images/results_table.png)

The best performing configuration achieved:

- **79.73% Test Accuracy**
- **79.56% Macro Precision**
- **79.59% Macro Recall**

using **tanh activation + momentum (α=0.95) + skip connections**.


## Training Behavior (Example Curve)

![Learning Curve](images/learning_curve.png)

The learning curves show stable convergence when momentum and skip connections are introduced, compared to unstable or near-random performance under sigmoid activations.


## Key Findings

- Sigmoid severely limits training in deeper networks.
- Tanh significantly improves gradient propagation.
- Momentum is critical for stable mini-batch optimization.
- Skip connections dramatically improve gradient flow and final accuracy.



## Implementation Constraints

- Manual softmax + cross-entropy
- Manual SGD updates
- Manual gradient resets
- No torch.optim usage



## Tech Stack

- PyTorch
- NumPy
- Matplotlib
- torchvision



## Reproducibility

All experiments were run with:
- Seed = 42
- Fixed train/test split
- CPU execution

---

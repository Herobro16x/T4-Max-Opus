# T4 MAX OPUS

# NAFM: Neuron Allocation & Filtering Mechanism

##  Overview

This project explores a simple approach to neural network compression under strict neuron constraints.

Goal:
How much can we reduce active neurons while maintaining performance?

Key ideas:
- Dynamic neuron selection
- Hard limits on active neurons
- Training strategies to improve compressed models

---

##  Method

### 1. Importance-Based Selection
Each neuron is scored using:
- Activation magnitude
- Output layer contribution

Top-K neurons are selected during training.

---

### 2. Hard Constraint (V9)
We enforce a fixed number of active neurons.

Example:
256 → 60 neurons (~76% reduction)

---

### 3. Training Strategies

V11 — Learn → Compress  
- Train full model first  
- Then apply sparsity  

V13 — Gradual + Recovery  
- Gradually reduce neurons  
- Allow extra training at final constraint  

---

##  Results

### MNIST

| Neurons | Accuracy |
|--------|---------|
| 256 (baseline) | ~98.2% |
| 60 | ~96–97% |

---

### Fashion-MNIST

| Model | Accuracy |
|------|---------|
| Baseline (256) | ~90.0% |
| V9 (K=60) | ~86.6% |
| V11 (K=60) | ~88.1% |
| V13 (K=60) | ~88.7% |

---

##  Key Observations

- Neural networks have redundancy
- Performance drops gradually under compression
- Training strategy matters a lot
- Recovery after compression improves results

---

##  How to Run

Install:

pip install torch torchvision

Run:

python nafm_v9.py  
python nafm_v11.py  
python nafm_v13.py  

---

##  Structure

NAFM/  
├── nafm_v9.py  
├── nafm_v11.py  
├── nafm_v13.py  
├── README.md  

---

##  Limitations

- Small datasets only
- Simple model
- No comparison with standard pruning methods yet

---

##  Future Work

- Test on bigger datasets (CIFAR-10)
- Improve importance scoring
- Compare with pruning methods
- Measure speed improvements

---

##  Summary

This project shows that neural networks can be compressed significantly while keeping strong performance if training is done properly.

---

##  Author

Independent experiment on neural network efficiency.

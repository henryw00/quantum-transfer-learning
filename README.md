# Quantum Transfer Learning for Image Classification

CS 3891/5891 Quantum Computing — Final Project, Spring 2026

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all experiments (classical baseline + 3 quantum configs)
python quantum_transfer_learning.py

# 3. Results saved to results/ folder:
#    - training_curves.png
#    - accuracy_comparison.png
#    - parameter_comparison.png
#    - experiment_results.json
```

## What it does

Replaces ResNet18's final classification layer with a variational quantum circuit
("dressed quantum circuit") and compares it against a classical baseline on the
ants vs. bees dataset.

### Experiments run automatically:
1. **Classical baseline** — ResNet18 + small linear head (4 hidden units)
2. **Quantum 4 qubits** — ResNet18 + 4-qubit dressed quantum circuit
3. **Quantum 2 qubits** — ResNet18 + 2-qubit dressed quantum circuit
4. **Quantum 4 qubits (noisy)** — Same as #2 with depolarizing noise (p=0.05)

## Configuration

Edit the `CONFIG` dict at the top of `quantum_transfer_learning.py`:
- `num_epochs`: Start with 3 for a quick test, use 10-30 for real results
- `n_qubits`: Default qubit count (experiments override this)
- `noise_strength`: Depolarizing noise probability

## References

- Mari et al. (2019), "Transfer learning in hybrid classical-quantum neural networks"
  https://arxiv.org/abs/1912.08278
- PennyLane tutorial: https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning/

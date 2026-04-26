"""
Quantum Transfer Learning for Image Classification
====================================================
CS 3891/5891 Quantum Computing - Final Project
Vanderbilt University, Spring 2026

Based on: Mari et al. (2019) - "Transfer learning in hybrid classical-quantum neural networks"
Tutorial: https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning/

This script implements:
1. Hybrid classical-quantum transfer learning (ResNet18 + dressed quantum circuit)
2. Classical-only baseline (ResNet18 + classical linear head)
3. Qubit count experiments (2 vs 4 qubits)
4. Noise simulation (depolarizing channel)
5. Automated plotting of training curves and comparisons
"""

import time
import os
import copy
import json
import urllib.request
import shutil
from datetime import datetime

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms

# PennyLane
import pennylane as qml
from pennylane import numpy as np

# Plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# OpenMP threads
os.environ["OMP_NUM_THREADS"] = "1"

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# CONFIGURATION - Edit these to run different experiments
# ============================================================
CONFIG = {
    "n_qubits": 4,              # Number of qubits (try 2 or 4)
    "q_depth": 6,               # Depth of quantum circuit (variational layers)
    "step": 0.0004,             # Learning rate
    "batch_size": 4,            # Batch size
    "num_epochs": 10,           # Training epochs (use 3 for quick test, 10-30 for real results)
    "gamma_lr_scheduler": 0.1,  # LR decay factor every 10 epochs
    "q_delta": 0.01,            # Initial spread of quantum weights
    "noise_strength": 0.0,      # Depolarizing noise (0.0 = no noise, try 0.01, 0.05, 0.1)
    "experiment_name": "default",
}

# ============================================================
# DATA LOADING
# ============================================================
def load_data(batch_size=4):
    """Download and prepare the Hymenoptera dataset (ants vs bees)."""
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    data_dir = "hymenoptera_data"
    if not os.path.exists(data_dir):
        print("Downloading dataset...")
        urllib.request.urlretrieve(
            "https://download.pytorch.org/tutorial/hymenoptera_data.zip",
            f"{data_dir}.zip"
        )
        shutil.unpack_archive(f"{data_dir}.zip")
        print("Dataset ready.")

    image_datasets = {
        x if x == "train" else "validation": datasets.ImageFolder(
            os.path.join(data_dir, x), data_transforms[x]
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "validation"]}
    class_names = image_datasets["train"].classes

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True
        )
        for x in ["train", "validation"]
    }

    print(f"Classes: {class_names}")
    print(f"Train size: {dataset_sizes['train']}, Val size: {dataset_sizes['validation']}")

    return dataloaders, dataset_sizes, class_names


# ============================================================
# QUANTUM CIRCUIT COMPONENTS
# ============================================================
def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates."""
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)

def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis."""
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)

def entangling_layer(nqubits):
    """Layer of CNOTs: even pairs, then odd pairs."""
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])


def make_quantum_circuit(n_qubits, q_depth, noise_strength=0.0):
    """Create a quantum device and circuit with optional noise."""

    if noise_strength > 0:
        dev = qml.device("default.mixed", wires=n_qubits)
    else:
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def quantum_net(q_input_features, q_weights_flat):
        q_weights = q_weights_flat.reshape(q_depth, n_qubits)

        # Start from |+> state
        H_layer(n_qubits)

        # Embed features
        RY_layer(q_input_features)

        # Variational layers
        for k in range(q_depth):
            entangling_layer(n_qubits)
            RY_layer(q_weights[k])

            # Add depolarizing noise after each variational layer
            if noise_strength > 0:
                for i in range(n_qubits):
                    qml.DepolarizingChannel(noise_strength, wires=i)

        # Measure
        exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
        return tuple(exp_vals)

    return quantum_net


# ============================================================
# DRESSED QUANTUM NET (Hybrid Model Head)
# ============================================================
class DressedQuantumNet(nn.Module):
    """
    Dressed quantum circuit: classical pre-net -> quantum circuit -> classical post-net.
    Maps 512 ResNet features -> 2 class outputs.
    """
    def __init__(self, n_qubits, q_depth, q_delta, noise_strength=0.0):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.pre_net = nn.Linear(512, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))
        self.post_net = nn.Linear(n_qubits, 2)
        self.quantum_net = make_quantum_circuit(n_qubits, q_depth, noise_strength)

    def forward(self, input_features):
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        q_out = torch.Tensor(0, self.n_qubits)
        q_out = q_out.to(device)
        for elem in q_in:
            q_out_elem = torch.hstack(
                self.quantum_net(elem, self.q_params)
            ).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        return self.post_net(q_out)


# ============================================================
# CLASSICAL BASELINE HEAD
# ============================================================
class ClassicalNet(nn.Module):
    """
    Classical linear head for fair comparison.
    Maps 512 ResNet features -> 2 class outputs via a small hidden layer.
    """
    def __init__(self, hidden_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, input_features):
        return self.net(input_features)


# ============================================================
# TRAINING LOOP
# ============================================================
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=10):
    """Train and evaluate a model, returning history."""
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"  Epoch {epoch+1}/{num_epochs}")

        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())

            print(f"    {phase:>10s} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "validation" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    elapsed = time.time() - start
    print(f"  Training complete in {elapsed//60:.0f}m {elapsed%60:.0f}s")
    print(f"  Best val accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    history["best_val_acc"] = best_acc.item()
    history["training_time"] = elapsed
    return model, history


# ============================================================
# MODEL BUILDERS
# ============================================================
def build_hybrid_model(n_qubits, q_depth, q_delta, noise_strength=0.0):
    """Build ResNet18 with a quantum classification head."""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = DressedQuantumNet(n_qubits, q_depth, q_delta, noise_strength)
    model = model.to(device)

    # Count trainable params
    n_params = sum(p.numel() for p in model.fc.parameters() if p.requires_grad)
    print(f"  Quantum head params: {n_params} (qubits={n_qubits}, depth={q_depth}, noise={noise_strength})")
    return model, n_params


def build_classical_model(hidden_size=4):
    """Build ResNet18 with a classical classification head."""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = ClassicalNet(hidden_size=hidden_size)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.fc.parameters() if p.requires_grad)
    print(f"  Classical head params: {n_params} (hidden_size={hidden_size})")
    return model, n_params


# ============================================================
# PLOTTING
# ============================================================
def plot_comparison(results, output_dir="results"):
    """Generate comparison plots from experiment results."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, res in results.items():
        axes[0].plot(res["train_loss"], label=f"{name} (train)", linestyle="--")
        axes[0].plot(res["val_loss"], label=f"{name} (val)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for name, res in results.items():
        axes[1].plot(res["train_acc"], label=f"{name} (train)", linestyle="--")
        axes[1].plot(res["val_acc"], label=f"{name} (val)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"  Saved training_curves.png")

    # --- Bar chart: best val accuracy ---
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    accs = [results[n]["best_val_acc"] for n in names]
    colors = plt.cm.Set2(range(len(names)))
    bars = ax.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("Model Comparison: Best Validation Accuracy")
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved accuracy_comparison.png")

    # --- Parameter efficiency ---
    fig, ax = plt.subplots(figsize=(8, 5))
    params = [results[n].get("n_params", 0) for n in names]
    ax.bar(names, params, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Trainable Parameters")
    ax.set_title("Parameter Count Comparison")
    for i, (bar_x, p) in enumerate(zip(ax.patches, params)):
        ax.text(bar_x.get_x() + bar_x.get_width()/2, bar_x.get_height() + 0.5,
                str(p), ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved parameter_comparison.png")


# ============================================================
# MAIN: RUN ALL EXPERIMENTS
# ============================================================
def main():
    print("=" * 60)
    print("QUANTUM TRANSFER LEARNING - EXPERIMENT SUITE")
    print("=" * 60)

    # Load data once
    dataloaders, dataset_sizes, class_names = load_data(CONFIG["batch_size"])
    criterion = nn.CrossEntropyLoss()
    results = {}

    # --- Experiment 1: Classical baseline ---
    print("\n[1/4] CLASSICAL BASELINE (hidden=4)")
    model_classical, n_params_c = build_classical_model(hidden_size=4)
    optimizer = optim.Adam(model_classical.fc.parameters(), lr=CONFIG["step"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=CONFIG["gamma_lr_scheduler"])
    _, hist = train_model(model_classical, dataloaders, dataset_sizes,
                          criterion, optimizer, scheduler, CONFIG["num_epochs"])
    hist["n_params"] = n_params_c
    results["Classical (h=4)"] = hist

    # --- Experiment 2: Quantum 4 qubits (no noise) ---
    print("\n[2/4] QUANTUM HYBRID (4 qubits, no noise)")
    model_q4, n_params_q4 = build_hybrid_model(4, CONFIG["q_depth"], CONFIG["q_delta"], 0.0)
    optimizer = optim.Adam(model_q4.fc.parameters(), lr=CONFIG["step"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=CONFIG["gamma_lr_scheduler"])
    _, hist = train_model(model_q4, dataloaders, dataset_sizes,
                          criterion, optimizer, scheduler, CONFIG["num_epochs"])
    hist["n_params"] = n_params_q4
    results["Quantum 4q"] = hist

    # --- Experiment 3: Quantum 2 qubits (no noise) ---
    print("\n[3/4] QUANTUM HYBRID (2 qubits, no noise)")
    model_q2, n_params_q2 = build_hybrid_model(2, CONFIG["q_depth"], CONFIG["q_delta"], 0.0)
    optimizer = optim.Adam(model_q2.fc.parameters(), lr=CONFIG["step"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=CONFIG["gamma_lr_scheduler"])
    _, hist = train_model(model_q2, dataloaders, dataset_sizes,
                          criterion, optimizer, scheduler, CONFIG["num_epochs"])
    hist["n_params"] = n_params_q2
    results["Quantum 2q"] = hist

    # --- Experiment 4: Quantum 4 qubits with noise ---
    print("\n[4/4] QUANTUM HYBRID (4 qubits, noise=0.05)")
    model_q4n, n_params_q4n = build_hybrid_model(4, CONFIG["q_depth"], CONFIG["q_delta"], 0.05)
    optimizer = optim.Adam(model_q4n.fc.parameters(), lr=CONFIG["step"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=CONFIG["gamma_lr_scheduler"])
    _, hist = train_model(model_q4n, dataloaders, dataset_sizes,
                          criterion, optimizer, scheduler, CONFIG["num_epochs"])
    hist["n_params"] = n_params_q4n
    results["Quantum 4q (noisy)"] = hist

    # --- Plot results ---
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    plot_comparison(results)

    # --- Save raw results ---
    with open("results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Saved experiment_results.json")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        print(f"  {name:25s}  Best Val Acc: {res['best_val_acc']:.4f}  "
              f"Params: {res['n_params']:5d}  Time: {res['training_time']:.0f}s")


if __name__ == "__main__":
    main()

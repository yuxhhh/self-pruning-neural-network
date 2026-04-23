"""
Self-Pruning Neural Network on CIFAR-10
========================================
This script implements a feed-forward neural network with learnable "gate" parameters
that allow the network to prune its own weights during training via L1 sparsity regularization.
 
Author: [Your Name]
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
 
# ─────────────────────────────────────────────────────────────
# PART 1: PrunableLinear Layer
# ─────────────────────────────────────────────────────────────
 
class PrunableLinear(nn.Module):
    """
    A custom linear layer augmented with learnable gate parameters.
 
    Each weight has a corresponding gate_score (scalar). A sigmoid is applied
    to each gate_score to produce a gate in (0, 1). The weight is then
    element-wise multiplied by the gate before the linear transformation.
 
    If a gate approaches 0, the corresponding weight is effectively pruned.
    Gradients flow through both `weight` and `gate_scores` via autograd.
    """
 
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
 
        # Standard weight and bias — same as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
 
        # Gate scores: same shape as weight, initialized near 0
        # so sigmoid(gate_scores) ≈ 0.5 at the start (neutral)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
 
        # Kaiming initialization for weights (good for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=0.01)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Convert gate_scores → gates ∈ (0, 1) via sigmoid
        gates = torch.sigmoid(self.gate_scores)
 
        # Step 2: Element-wise multiply weights with gates
        pruned_weights = self.weight * gates
 
        # Step 3: Standard linear transformation (gradients flow to both weight & gate_scores)
        return F.linear(x, pruned_weights, self.bias)
 
    def get_gates(self) -> torch.Tensor:
        """Returns the actual gate values (after sigmoid) as a detached tensor."""
        return torch.sigmoid(self.gate_scores).detach()
 
    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of gate values (encourages them to go to 0)."""
        return torch.sigmoid(self.gate_scores).abs().sum()
 
 
# ─────────────────────────────────────────────────────────────
# Neural Network using PrunableLinear
# ─────────────────────────────────────────────────────────────
 
class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 classification.
    Uses PrunableLinear layers instead of standard nn.Linear.
 
    Architecture:
        Input (3072) → 512 → 256 → 128 → 10 (classes)
    """
 
    def __init__(self):
        super().__init__()
        # CIFAR-10: 32x32x3 = 3072 input features
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 10)
 
        self.dropout = nn.Dropout(p=0.3)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)          # Flatten: (B, 3, 32, 32) → (B, 3072)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
 
    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum of L1 gate norms across all PrunableLinear layers."""
        loss = torch.tensor(0.0)
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                loss = loss + module.sparsity_loss()
        return loss
 
    def get_all_gates(self) -> np.ndarray:
        """Collect all gate values from all layers as a flat numpy array."""
        gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(module.get_gates().cpu().numpy().flatten())
        return np.concatenate(gates)
 
    def compute_sparsity_level(self, threshold: float = 1e-2) -> float:
        """
        Fraction of weights whose gate value is below `threshold`.
        A high value means the network has pruned most of its connections.
        """
        all_gates = self.get_all_gates()
        pruned = np.sum(all_gates < threshold)
        return pruned / len(all_gates) * 100.0
 
 
# ─────────────────────────────────────────────────────────────
# PART 3: Data Loading
# ─────────────────────────────────────────────────────────────
 
def get_dataloaders(batch_size: int = 128):
    """Download and prepare CIFAR-10 train and test loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),   # CIFAR-10 mean
                             (0.2470, 0.2435, 0.2616)),  # CIFAR-10 std
    ])
 
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform)
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
 
    return train_loader, test_loader
 
 
# ─────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────
 
def train_one_epoch(model, loader, optimizer, lambda_sparse, device):
    """Train for one epoch and return average total loss."""
    model.train()
    total_loss = 0.0
 
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
 
        optimizer.zero_grad()
 
        # Forward pass
        logits = model(images)
 
        # Classification loss (Cross-Entropy)
        classification_loss = F.cross_entropy(logits, labels)
 
        # Sparsity regularization loss (L1 on all gates)
        sparsity_loss = model.total_sparsity_loss().to(device)
 
        # Total loss
        loss = classification_loss + lambda_sparse * sparsity_loss
 
        # Backward pass — gradients flow to both weight and gate_scores
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
 
    return total_loss / len(loader)
 
 
def evaluate(model, loader, device):
    """Evaluate model accuracy on a data loader."""
    model.eval()
    correct = 0
    total   = 0
 
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
 
    return correct / total * 100.0
 
 
# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────
 
def plot_gate_distribution(model, lambda_val, save_path="gate_distribution.png"):
    """
    Plot histogram of final gate values.
    A good result shows a large spike at 0 (pruned) and a cluster near 1 (active).
    """
    gates = model.get_all_gates()
 
    plt.figure(figsize=(8, 5))
    plt.hist(gates, bins=80, color='steelblue', edgecolor='white', linewidth=0.4)
    plt.title(f"Gate Value Distribution  (λ = {lambda_val})", fontsize=14, fontweight='bold')
    plt.xlabel("Gate Value (after sigmoid)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.axvline(x=0.01, color='red', linestyle='--', linewidth=1.2, label='Prune threshold (0.01)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot saved] → {save_path}")
 
 
# ─────────────────────────────────────────────────────────────
# Main Experiment: Compare 3 λ values
# ─────────────────────────────────────────────────────────────
 
def run_experiment(lambda_sparse, num_epochs, train_loader, test_loader, device):
    """Train one model with a given lambda and return accuracy + sparsity."""
    print(f"\n{'='*55}")
    print(f"  Training with λ = {lambda_sparse}")
    print(f"{'='*55}")
 
    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
 
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, lambda_sparse, device)
        scheduler.step()
 
        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, test_loader, device)
            sparsity = model.compute_sparsity_level()
            print(f"  Epoch {epoch:3d}/{num_epochs} | Loss: {avg_loss:.4f} | "
                  f"Test Acc: {acc:.2f}% | Sparsity: {sparsity:.1f}%")
 
    final_acc     = evaluate(model, test_loader, device)
    final_sparsity = model.compute_sparsity_level()
 
    print(f"\n  ✓ Final Test Accuracy : {final_acc:.2f}%")
    print(f"  ✓ Final Sparsity Level: {final_sparsity:.1f}%")
 
    return model, final_acc, final_sparsity
 
 
def main():
    # ── Device ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    # ── Hyperparameters ──────────────────────────────────────
    BATCH_SIZE  = 128
    NUM_EPOCHS  = 30   # Increase to 50–80 for better results if time allows
 
    # Three lambda values to compare (low / medium / high sparsity pressure)
    LAMBDAS = [1e-4, 1e-3, 5e-3]
 
    # ── Data ─────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
 
    # ── Run experiments ──────────────────────────────────────
    results = []
    best_model, best_lambda = None, None
    best_score = -1
 
    for lam in LAMBDAS:
        model, acc, sparsity = run_experiment(
            lambda_sparse=lam,
            num_epochs=NUM_EPOCHS,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device
        )
        results.append((lam, acc, sparsity))
 
        # Track "best" model: highest accuracy with reasonable sparsity
        if acc > best_score:
            best_score  = acc
            best_model  = model
            best_lambda = lam
 
    # ── Summary Table ─────────────────────────────────────────
    print("\n\n" + "="*55)
    print("  RESULTS SUMMARY")
    print("="*55)
    print(f"  {'Lambda':<12} {'Test Accuracy':>15} {'Sparsity (%)':>15}")
    print(f"  {'-'*42}")
    for lam, acc, sparsity in results:
        print(f"  {lam:<12} {acc:>14.2f}% {sparsity:>14.1f}%")
    print("="*55)
 
    # ── Plot gate distribution for best model ─────────────────
    plot_gate_distribution(best_model, best_lambda, save_path="gate_distribution.png")
 
    # ── Save results to text file ─────────────────────────────
    with open("results_summary.txt", "w") as f:
        f.write("Self-Pruning Neural Network — Results Summary\n")
        f.write("="*50 + "\n")
        f.write(f"{'Lambda':<12} {'Test Accuracy':>15} {'Sparsity (%)':>15}\n")
        f.write("-"*42 + "\n")
        for lam, acc, sparsity in results:
            f.write(f"{lam:<12} {acc:>14.2f}% {sparsity:>14.1f}%\n")
    print("\n  [Results saved] → results_summary.txt")
 
 
if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------
# 1. Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64,   shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=1000, shuffle=False)

# -------------------------
# 2. Simple CNN Model
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self):                        # FIX 1: was "def init"
        super(SimpleCNN, self).__init__()      # FIX 2: was ".init()"
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1   = nn.Linear(32 * 24 * 24, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model     = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# -------------------------
# 3. Train Baseline Model
# -------------------------
def train(model, loader, epochs=2):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# -------------------------
# 4. Evaluate Model
# -------------------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs     = model(images)
            predictions = outputs.argmax(dim=1)
            correct    += (predictions == labels).sum().item()
            total      += labels.size(0)
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# -------------------------
# 5. FGSM Adversarial Attack
# -------------------------
def fgsm_attack(image, epsilon, gradient):
    perturbation      = epsilon * gradient.sign()
    adversarial_image = image + perturbation
    adversarial_image = torch.clamp(adversarial_image, 0, 1)
    return adversarial_image

def evaluate_fgsm(model, loader, epsilon=0.25):
    model.eval()
    correct      = 0
    total        = 0
    sample_clean = None
    sample_adv   = None

    for images, labels in loader:
        images, labels       = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        loss    = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        gradient           = images.grad.data
        adversarial_images = fgsm_attack(images, epsilon, gradient)

        with torch.no_grad():
            adversarial_outputs = model(adversarial_images)
            predictions         = adversarial_outputs.argmax(dim=1)
            correct            += (predictions == labels).sum().item()
            total              += labels.size(0)

        if sample_clean is None:
            sample_clean = images.detach().cpu()
            sample_adv   = adversarial_images.detach().cpu()

    accuracy = correct / total
    print(f"FGSM Accuracy with epsilon={epsilon}: {accuracy * 100:.2f}%")
    return accuracy, sample_clean, sample_adv

# -------------------------
# 6. Simple Defense: Adversarial Training
# -------------------------
def adversarial_train(model, loader, epochs=2, epsilon=0.25):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels       = images.to(device), labels.to(device)
            images.requires_grad = True

            outputs = model(images)
            loss    = criterion(outputs, labels)

            model.zero_grad()
            loss.backward()

            gradient           = images.grad.data
            adversarial_images = fgsm_attack(images, epsilon, gradient)

            optimizer.zero_grad()
            defended_outputs = model(adversarial_images.detach())
            defended_loss    = criterion(defended_outputs, labels)

            defended_loss.backward()
            optimizer.step()

            total_loss += defended_loss.item()

        print(f"Defense Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# -------------------------
# 7. Graphs
# -------------------------
def plot_examples(clean, adv, epsilon, n=8):
    fig, axes = plt.subplots(2, n, figsize=(14, 4))
    fig.suptitle(f"Clean vs Adversarial (epsilon={epsilon})")
    for i in range(n):
        axes[0, i].imshow(clean[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(adv[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Clean",       fontsize=9)
    axes[1, 0].set_title("Adversarial", fontsize=9)
    plt.tight_layout()
    plt.savefig("sample_adversarial_examples.png", dpi=150)
    plt.close()
    print("Saved: sample_adversarial_examples.png")

def plot_epsilon_curve(epsilons, base_accs, def_accs):
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, [a * 100 for a in base_accs], "o-r", label="Baseline")
    plt.plot(epsilons, [a * 100 for a in def_accs],  "s-b", label="Defended")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epsilon Under FGSM Attack")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_vs_epsilon.png", dpi=150)
    plt.close()
    print("Saved: accuracy_vs_epsilon.png")

def plot_comparison(clean_acc, attack_acc, def_clean, def_attack):
    labels   = ["Clean", "Under Attack (e=0.25)"]
    baseline = [clean_acc * 100, attack_acc * 100]
    defended = [def_clean * 100, def_attack * 100]
    x     = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - width/2 for i in x], baseline, width, label="Baseline", color="tomato")
    ax.bar([i + width/2 for i in x], defended, width, label="Defended", color="steelblue")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Baseline vs Defended Model")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("defense_comparison.png", dpi=150)
    plt.close()
    print("Saved: defense_comparison.png")

# -------------------------
# 8. Main
# -------------------------
if __name__ == "__main__":

    # Phase 1 - train baseline
    print("\n=== Training Baseline ===")
    train(model, train_loader, epochs=2)
    clean_acc = evaluate(model, test_loader)

    # Phase 2 - attack
    print("\n=== Running FGSM Attack ===")
    epsilons   = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    base_accs  = []
    clean_imgs = adv_imgs = None

    for eps in epsilons:
        acc, c, a = evaluate_fgsm(model, test_loader, epsilon=eps)
        base_accs.append(acc)
        if eps == 0.25:
            clean_imgs, adv_imgs = c, a

    attack_acc = base_accs[epsilons.index(0.25)]
    plot_examples(clean_imgs, adv_imgs, epsilon=0.25)

    # Phase 3 - defense
    print("\n=== Adversarial Training Defense ===")
    defense_model = SimpleCNN().to(device)
    optimizer     = optim.Adam(defense_model.parameters(), lr=0.001)
    adversarial_train(defense_model, train_loader, epochs=2, epsilon=0.25)
    def_clean_acc = evaluate(defense_model, test_loader)

    def_accs = []
    for eps in epsilons:
        acc, _, _ = evaluate_fgsm(defense_model, test_loader, epsilon=eps)
        def_accs.append(acc)

    def_attack_acc = def_accs[epsilons.index(0.25)]

    # Phase 4 - save graphs
    print("\n=== Saving Graphs ===")
    plot_epsilon_curve(epsilons, base_accs, def_accs)
    plot_comparison(clean_acc, attack_acc, def_clean_acc, def_attack_acc)

    # Results summary
    print("\n========== RESULTS ==========")
    print(f"Baseline clean accuracy:        {clean_acc * 100:.2f}%")
    print(f"Baseline under attack (e=0.25): {attack_acc * 100:.2f}%")
    print(f"Defended clean accuracy:        {def_clean_acc * 100:.2f}%")
    print(f"Defended under attack (e=0.25): {def_attack_acc * 100:.2f}%")
    print(f"Improvement under attack:       +{(def_attack_acc - attack_acc) * 100:.2f}%")
    print("=============================")

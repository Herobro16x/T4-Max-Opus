import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# DATA
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True
)

# MODEL
class NAFM_V9(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, k_active=60):
        super().__init__()
        self.hidden_size = hidden_size
        self.k_active = k_active

        self.weights = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.out = nn.Linear(hidden_size, 10)

        self.register_buffer("importance", torch.zeros(hidden_size))
        self.last_active = hidden_size

    def forward(self, x):
        batch_size = x.shape[0]
        full_hidden = torch.relu(torch.matmul(x, self.weights))

        out_weights = torch.abs(self.out.weight).mean(dim=0)
        importance_score = full_hidden.mean(dim=0) * out_weights
        self.importance = 0.9 * self.importance + 0.1 * importance_score.detach()

        _, idx = torch.topk(self.importance, self.k_active)

        self.last_active = len(idx)

        hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
        hidden[:, idx] = torch.relu(torch.matmul(x, self.weights[:, idx]))

        return self.out(hidden)

# TRAIN
def train(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        correct = 0
        total = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        print(f"Epoch {epoch} | Acc: {correct/total:.4f} | Active: {model.last_active}")

# RUN
if __name__ == "__main__":
    print("=== NAFM V9 ===")
    model = NAFM_V9(k_active=60)
    train(model)

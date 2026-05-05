# Same imports + data

class NAFM_V13(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.current_k = hidden_size

        self.weights = nn.Parameter(torch.randn(784, hidden_size) * 0.01)
        self.out = nn.Linear(hidden_size, 10)

        self.register_buffer("importance", torch.zeros(hidden_size))
        self.last_active = hidden_size

    def set_k(self, k):
        self.current_k = k

    def forward(self, x):
        full_hidden = torch.relu(torch.matmul(x, self.weights))

        out_weights = torch.abs(self.out.weight).mean(dim=0)
        importance_score = full_hidden.mean(dim=0) * out_weights
        self.importance = 0.9 * self.importance + 0.1 * importance_score.detach()

        _, idx = torch.topk(self.importance, self.current_k)

        self.last_active = len(idx)

        hidden = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        hidden[:, idx] = torch.relu(torch.matmul(x, self.weights[:, idx]))

        return self.out(hidden)

def train():
    model = NAFM_V13()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    schedule = [256, 256, 256, 180, 120, 60, 60, 60]

    for epoch in range(8):
        model.set_k(schedule[epoch])
        print(f"\nEpoch {epoch} | K={schedule[epoch]}")

        correct = total = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        print(f"Acc: {correct/total:.4f} | Active: {model.last_active}")

if __name__ == "__main__":
    print("=== NAFM V13 ===")
    train()

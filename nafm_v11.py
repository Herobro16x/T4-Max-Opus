# Same imports + data as V9

class NAFM_V11(nn.Module):
    def __init__(self, hidden_size=256, k_active=60):
        super().__init__()
        self.hidden_size = hidden_size
        self.k_active = k_active
        self.sparsity_on = False

        self.weights = nn.Parameter(torch.randn(784, hidden_size) * 0.01)
        self.out = nn.Linear(hidden_size, 10)

        self.register_buffer("importance", torch.zeros(hidden_size))
        self.last_active = hidden_size

    def forward(self, x):
        full_hidden = torch.relu(torch.matmul(x, self.weights))

        out_weights = torch.abs(self.out.weight).mean(dim=0)
        importance_score = full_hidden.mean(dim=0) * out_weights
        self.importance = 0.9 * self.importance + 0.1 * importance_score.detach()

        if not self.sparsity_on:
            self.last_active = self.hidden_size
            return self.out(full_hidden)

        _, idx = torch.topk(self.importance, self.k_active)

        self.last_active = len(idx)

        hidden = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        hidden[:, idx] = torch.relu(torch.matmul(x, self.weights[:, idx]))

        return self.out(hidden)

def train():
    model = NAFM_V11()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(6):
        if epoch == 3:
            print("\n--- ACTIVATING SPARSITY ---\n")
            model.sparsity_on = True

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

        print(f"Epoch {epoch} | Acc: {correct/total:.4f} | Active: {model.last_active}")

if __name__ == "__main__":
    print("=== NAFM V11 ===")
    train()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, z):
        return self.net(z)


def augment_minority_classes(X, y, target_class_idx, n_samples_to_generate=100):
    X_target = X[y == target_class_idx]
    if len(X_target) == 0:
        return X, y

    feature_dim = X.shape[1]
    noise_dim = 16

    gen = Generator(noise_dim, feature_dim)
    optimizer = optim.Adam(gen.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_tensor = torch.FloatTensor(X_target)
    for _ in range(50):
        z = torch.randn(X_target.shape[0], noise_dim)
        fake_data = gen(z)
        loss = criterion(fake_data, X_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    z = torch.randn(n_samples_to_generate, noise_dim)
    synthetic_samples = gen(z).detach().numpy()
    synthetic_labels = np.full(n_samples_to_generate, target_class_idx)

    X_augmented = np.vstack([X, synthetic_samples])
    y_augmented = np.concatenate([y, synthetic_labels])

    return X_augmented, y_augmented
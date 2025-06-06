import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(42)
np.random.seed(42)

# Load and shuffle the dataset
df = pd.read_csv("mitbih_ecg_processed.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop rows with missing labels
df = df.dropna(subset=['label'])

# Split features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert training data to DataFrame for GAN input
train_df = pd.DataFrame(X_train)
train_df['label'] = y_train.reset_index(drop=True)

# Filter minority classes
minority_classes = [1, 2, 3, 4]
minority_df = train_df[train_df['label'].isin(minority_classes)]

# Prepare GAN training data
X_minority = minority_df.drop('label', axis=1).values
y_minority = minority_df['label'].values

# Torch Dataset and Dataloader
import torch
from torch.utils.data import Dataset, DataLoader

class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

data_loader = DataLoader(ECGDataset(X_minority, y_minority), batch_size=64, shuffle=True)

# GAN Model Definition
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

noise_dim = 100
input_dim = X_minority.shape[1]

gen = Generator(noise_dim, input_dim)
disc = Discriminator(input_dim)

criterion = nn.BCELoss()
g_optimizer = torch.optim.Adam(gen.parameters(), lr=0.0002)
d_optimizer = torch.optim.Adam(disc.parameters(), lr=0.0002)

gen.train()
disc.train()

# Train GAN
for epoch in range(1000):
    for real_data, _ in data_loader:
        batch_size = real_data.size(0)

        # Real and fake labels
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train discriminator
        d_optimizer.zero_grad()
        outputs = disc(real_data)
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(batch_size, noise_dim)
        fake_data = gen(z)
        outputs = disc(fake_data.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # Train generator
        g_optimizer.zero_grad()
        z = torch.randn(batch_size, noise_dim)
        fake_data = gen(z)
        outputs = disc(fake_data)
        g_loss = criterion(outputs, real_labels)

        g_loss.backward()
        g_optimizer.step()

    if (epoch+1) % 200 == 0:
        print(f"Epoch {epoch+1}/1000, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Generate synthetic samples per class (e.g., class 1, 2, 3, 4)
synthetic_samples = []
labels_to_generate = [1, 2, 3, 4]

samples_per_class = {1:72286 , 2: 72286, 3: 72286, 4: 72286}

for label in labels_to_generate:
    n_samples = samples_per_class[label]
    z = torch.randn(n_samples, noise_dim)
    synth = gen(z).detach().numpy()
    synth_df = pd.DataFrame(synth)
    synth_df['label'] = label
    synthetic_samples.append(synth_df)

# Combine all synthetic samples
synthetic_df = pd.concat(synthetic_samples, axis=0).reset_index(drop=True)

# Combine with real majority class (class 0)
real_majority_df = train_df[train_df['label'] == 0]
balanced_df = pd.concat([real_majority_df, synthetic_df], axis=0).reset_index(drop=True)

# Final checks
print(f"Balanced training data shape: {balanced_df.shape}")
assert balanced_df.shape[1] == 188, "Expected 187 features and 1 label column."
assert balanced_df.isnull().sum().sum() == 0, "Missing values found in balanced data."

# Save balanced training data before PCA
balanced_df.to_csv("Balanced_Training_Data_Pre_PCA.csv", index=False)
print("Balanced training data saved as 'Balanced_Training_Data_Pre_PCA.csv'")

# PCA on training and test data
X_balanced = balanced_df.iloc[:, :-1]
y_balanced = balanced_df['label']

X_test_df = pd.DataFrame(X_test)
X_test_df['label'] = y_test.reset_index(drop=True)

pca = PCA(n_components=30)
X_bal_pca = pca.fit_transform(X_balanced)
X_test_pca = pca.transform(X_test)

# Save PCA-reduced training and test data
pd.DataFrame(X_bal_pca).assign(label=y_balanced.reset_index(drop=True)).to_csv("Training_PCA.csv", index=False)
pd.DataFrame(X_test_pca).assign(label=y_test.reset_index(drop=True)).to_csv("Testing_PCA.csv", index=False)

print("PCA reduced training and test data saved as 'Training_PCA.csv' and 'Testing_PCA.csv'")
# Plot label distribution
def plot_label_distribution(y, title):
    count = pd.Series(y).value_counts().reindex([0, 1, 2, 3, 4], fill_value=0)
    plt.bar(count.index, count.values, color='lightblue', edgecolor='black')
    for i, v in enumerate(count.values):
        plt.text(i, v + 5, str(v), ha='center')
    plt.xlabel("Class Labels")
    plt.ylabel("Samples")
    plt.title(title)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_label_distribution(y_balanced, "Training Labels Distribution")
plt.subplot(1, 2, 2)
plot_label_distribution(y_test, "Test Labels Distribution")
plt.tight_layout()
plt.show()

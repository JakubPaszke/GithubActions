import pandas as pd
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
import sys

# Parametry przekazywane z workflow
epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01

# ----- Wczytaj dane -----
def custom_reader(row):
    parts = row.strip().split('\t')
    label = parts[0]
    text = '\t'.join(parts[1:])
    return label, text

with open('train/train.tsv', encoding='utf-8') as f:
    data = [custom_reader(line) for line in f if line.strip()]

train = pd.DataFrame(data, columns=['label', 'text'])

# ----- Wczytaj embeddings -----
w2v_model = KeyedVectors.load('word2vec_100_3_polish.bin')
vector_size = w2v_model.vector_size

# ----- Funkcje pomocnicze -----
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-ząćęłńóśźż ]', '', text)
    return text

def text_to_vec(text, model, vector_size):
    words = clean_text(text).split()
    vectors = [model[w] for w in words if w in model]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

# ----- Przygotuj dane do trenowania -----
X = np.vstack([text_to_vec(t, w2v_model, vector_size) for t in train['text']])
y = train['label'].values

# ----- Dataset + DataLoader -----
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TextDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ----- Prosty model -----
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)
    def forward(self, x):
        return self.fc(x)

num_classes = len(set(y))
model = SimpleClassifier(vector_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ----- Trening -----
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

# ----- Zapisz model -----
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")

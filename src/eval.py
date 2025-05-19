import pandas as pd
import re
import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors

# ----- Wczytaj dane -----
test = pd.read_csv('dev-0/in.tsv', sep='\t', header=None, names=['text'])
test_labels = pd.read_csv('dev-0/expected.tsv', sep='\t', header=None, names=['label'])['label'].values

# ----- Wczytaj embeddings -----
w2v_model = KeyedVectors.load('word2vec_polish.bin')
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

# ----- Przygotuj dane -----
X_test = np.vstack([text_to_vec(t, w2v_model, vector_size) for t in test['text']])

# ----- Klasa modelu -----
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)
    def forward(self, x):
        return self.fc(x)

num_classes = len(set(test_labels))  # UWAGA: musi być zgodne z train.py
model = SimpleClassifier(vector_size, num_classes)
model.load_state_dict(torch.load("model.pth"))
model.eval()

with torch.no_grad():
    inputs = torch.tensor(X_test, dtype=torch.float32)
    outputs = model(inputs)
    preds = torch.argmax(outputs, dim=1).numpy()

# ----- Ewaluacja -----
accuracy = np.mean(preds == test_labels)
print(f"Accuracy on dev-0: {accuracy:.4f}")

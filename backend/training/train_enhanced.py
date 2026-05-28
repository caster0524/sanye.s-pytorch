"""
Enhanced training script for classification and NLP sentiment
Uses larger, more diverse datasets for better accuracy
"""
import os
import sys
import json
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets

import numpy as np
from collections import Counter

BASE_DIR = Path(__file__).parent.parent.parent  # pytorch-web-app
DATA_DIR = BASE_DIR / "public" / "datasets"
MODEL_DIR = BASE_DIR / "public" / "models" / "trained"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def train_classification():
    """Train ResNet50 classifier on enhanced dataset"""
    print("\n" + "=" * 60)
    print("Training Image Classification Model (ResNet50)")
    print("=" * 60)

    dataset_path = DATA_DIR / "classification_enhanced"
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder(str(dataset_path), transform=train_transform)
    num_classes = len(full_dataset.classes)
    print(f"Dataset: {len(full_dataset)} images, {num_classes} classes")
    print(f"Classes: {full_dataset.classes}")

    # Check class distribution
    class_counts = Counter()
    for _, label in full_dataset.samples:
        class_counts[label] += 1
    for idx, count in sorted(class_counts.items()):
        print(f"  {full_dataset.classes[idx]}: {count}")

    # Split: 80% train, 20% val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                               generator=torch.Generator().manual_seed(42))
    # Use val transform for validation
    full_dataset_val = datasets.ImageFolder(str(dataset_path), transform=val_transform)
    _, val_dataset = random_split(full_dataset_val, [train_size, val_size],
                                   generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"Train: {train_size}, Val: {val_size}")

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(50):
        model.train()
        train_loss = 0.0
        train_correct = 0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        train_loss /= train_size
        train_acc = 100.0 * train_correct / train_size

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_size
        val_acc = 100.0 * val_correct / val_size
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:2d}/50 | Train: {train_acc:.1f}% Loss: {train_loss:.4f} | "
              f"Val: {val_acc:.1f}% Loss: {val_loss:.4f} | {elapsed:.1f}s")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_path = MODEL_DIR / "resnet50_classifier.pth"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': full_dataset.classes,
                'val_accuracy': val_acc,
                'num_classes': num_classes,
                'epoch': epoch + 1,
            }, model_path)
            print(f"  -> Saved best model (val_acc={val_acc:.1f}%)")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.1f}%")

    # Final test on all data
    model.eval()
    all_correct = 0
    all_samples = 0
    test_loader = DataLoader(full_dataset_val, batch_size=32, shuffle=False)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_correct += predicted.eq(labels).sum().item()
            all_samples += labels.size(0)

    final_acc = 100.0 * all_correct / all_samples
    print(f"Final accuracy on full dataset: {final_acc:.1f}%")
    return final_acc


def train_nlp_sentiment():
    """Train NLP sentiment classifier on enhanced dataset"""
    print("\n" + "=" * 60)
    print("Training NLP Sentiment Classification Model")
    print("=" * 60)

    dataset_path = DATA_DIR / "nlp_enhanced" / "sentiment.json"
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    print(f"Dataset: {len(texts)} samples")

    # Count distribution
    label_names = {0: "negative", 1: "positive", 2: "neutral"}
    for label_id, name in label_names.items():
        count = sum(1 for l in labels if l == label_id)
        print(f"  {name}: {count}")

    # Build vocabulary from training data
    from collections import Counter as Ctr
    import re

    # Preprocess
    def preprocess(text):
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', '', text)
        text = text.lower().strip()
        return text

    # Split
    indices = list(range(len(texts)))
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(0.8 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]

    # Build vocab from training data only
    all_words = []
    for i in train_idx:
        words = preprocess(texts[i]).split()
        all_words.extend(words)

    word_counts = Ctr(all_words)
    # Keep top 3000 words
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.most_common(3000))}
    vocab_size = len(vocab) + 1  # +1 for padding
    print(f"Vocabulary size: {vocab_size}")

    # Convert texts to vectors
    def text_to_vector(text, max_len=100):
        words = preprocess(text).split()
        vec = np.zeros(max_len, dtype=np.int64)
        unsafe_idx = []
        for i, word in enumerate(words[:max_len]):
            if word in vocab:
                vec[i] = vocab[word]
            else:
                unsafe_idx.append(i)
        return vec

    # Prepare data
    X_train = np.stack([text_to_vector(texts[i]) for i in train_idx])
    y_train = np.array([labels[i] for i in train_idx])
    X_val = np.stack([text_to_vector(texts[i]) for i in val_idx])
    y_val = np.array([labels[i] for i in val_idx])

    # Also use TF-IDF features for better accuracy
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        token_pattern=r'(?u)\b\w+\b',
    )
    X_train_tfidf = tfidf.fit_transform([texts[i] for i in train_idx])
    X_val_tfidf = tfidf.transform([texts[i] for i in val_idx])

    # PyTorch model
    class SentimentClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim=128, num_classes=3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(embedding_dim, 128, num_layers=2,
                                bidirectional=True, batch_first=True, dropout=0.3)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(128 * 2, num_classes)  # bidirectional

        def forward(self, x):
            x = self.embedding(x)
            _, (hidden, _) = self.lstm(x)
            # Use last layer hidden states from both directions
            x = torch.cat((hidden[-2], hidden[-1]), dim=1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    model = SentimentClassifier(vocab_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    best_val_acc = 0.0

    for epoch in range(30):
        model.train()
        train_loss = 0.0
        train_correct = 0
        t0 = time.time()

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(y_batch).sum().item()

        train_loss /= len(train_idx)
        train_acc = 100.0 * train_correct / len(train_idx)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * x_batch.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(y_batch).sum().item()

        val_loss /= len(val_idx)
        val_acc = 100.0 * val_correct / len(val_idx)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:2d}/30 | Train: {train_acc:.1f}% Loss: {train_loss:.4f} | "
              f"Val: {val_acc:.1f}% Loss: {val_loss:.4f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = MODEL_DIR / "nlp_sentiment.pth"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'vocab_size': vocab_size,
                'val_accuracy': val_acc,
                'label_names': label_names,
            }, model_path)

    print(f"\nNLP training complete! Best val accuracy: {best_val_acc:.1f}%")
    return best_val_acc


if __name__ == "__main__":
    t_start = time.time()

    print("Enhanced Model Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model output dir: {MODEL_DIR}")

    # Train classification
    cls_acc = train_classification()

    # Train NLP
    nlp_acc = train_nlp_sentiment()

    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"All training complete! Total time: {total_time / 60:.1f} minutes")
    if cls_acc:
        print(f"Classification accuracy: {cls_acc:.1f}%")
    if nlp_acc:
        print(f"NLP sentiment accuracy: {nlp_acc:.1f}%")
"""
BERT-based Phishing Detection Training Script

- Trains a BERT model on email/SMS data
- Evaluates performance using accuracy, AUC, confusion matrix, ROC curve
- Saves trained model and tokenizer locally

NOTE:
Dataset is not included in the repository due to size/privacy reasons.
Expected CSV columns:
- subject
- body
- label (0 = ham, 1 = phishing)
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    roc_curve
)

import tensorflow as tf
from transformers import (
    BertTokenizerFast,
    TFBertForSequenceClassification,
    create_optimizer
)

# === Load and Clean Data ===
# Dataset not included in repo due to size/privacy constraints
df = pd.read_csv("data/Book1.csv", low_memory=False, dtype={'label': 'string'}, on_bad_lines='skip')

# Ensure required columns exist
for col in ['subject', 'body', 'label']:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df['subject'] = df['subject'].fillna('')
df['body'] = df['body'].fillna('')
df['message'] = (df['subject'] + ' ' + df['body']).str.strip()
df = df[df['message'].str.len() > 0].copy()

# Standardize labels
df['label'] = df['label'].str.strip().str.lower()
label_map = {
    'ham': 0, 'not spam': 0, 'legit': 0, 'safe': 0, '0': 0, 'no': 0,
    'spam': 1, 'phishing': 1, 'phish': 1, 'malicious': 1, '1': 1, 'yes': 1
}
df['label'] = df['label'].map(label_map)
df.dropna(subset=['label'], inplace=True)
df['label'] = df['label'].astype(int)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# === Tokenization ===
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(
    list(X_train),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="tf"
)

test_encodings = tokenizer(
    list(X_test),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="tf"
)

train_dataset = (
    tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train))
    .shuffle(1000)
    .batch(16)
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (dict(test_encodings), y_test)
).batch(16)

# === Build Model ===
bert_model = TFBertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# === Optimizer ===
steps_per_epoch = len(X_train) // 16
num_train_steps = steps_per_epoch * 5

optimizer, _ = create_optimizer(
    init_lr=3e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps
)

bert_model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# === Callbacks ===
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "bert_best_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)

# === Train ===
history = bert_model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# === Save Model & Tokenizer ===
save_dir = "bert_model_output"
os.makedirs(save_dir, exist_ok=True)

bert_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# === Evaluation ===
preds = bert_model.predict(test_dataset).logits
probs = tf.nn.softmax(preds, axis=1)[:, 1].numpy()
pred_labels = np.argmax(preds, axis=1)

print("Confusion Matrix:\n", confusion_matrix(y_test, pred_labels))
print("\nAccuracy:", accuracy_score(y_test, pred_labels))
print("\nAUC:", roc_auc_score(y_test, probs))
print("\nClassification Report:\n", classification_report(y_test, pred_labels))

# === Accuracy & Loss Curves ===
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# === Confusion Matrix ===
cm = confusion_matrix(y_test, pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, probs):.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

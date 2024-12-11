import transformers
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Suppress symlink warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# ignore the warning as long as you're training the model properly
transformers.logging.set_verbosity_error()

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Sample data
texts = [
    # Positive reviews
    "I absolutely love this!", "Best product I've ever used.", "Such a wonderful experience.", 
    "The quality is outstanding.", "Exceeded all my expectations.", 
    "Highly recommend this to everyone.", "A masterpiece in every sense.", 
    "This is my favorite purchase so far.", "Fantastic service, would buy again.", 
    "Beautiful design and functionality.", "Incredible performance and features.", 
    "This movie had me in tears of joy.", "Great job by the entire cast.", 
    "The storyline was compelling and heartwarming.", "An excellent addition to the franchise.", 

    # Negative reviews
    "Worst product ever.", "I hated every second of it.", 
    "Completely unacceptable quality.", "Do not waste your money on this.", 
    "This is an absolute disaster.", "It broke after one use.", 
    "Terrible customer service.", "Unbelievably bad experience.", 
    "The movie was dull and predictable.", "I regret watching it.", 
    "The characters were flat and uninspiring.", "A total waste of time.", 
    "The plot was full of holes and made no sense.", 
    "This was a major disappointment.", 

    # Neutral/mixed reviews
    "It's okay, not great but not bad either.", 
    "The product is fine but could be better.", 
    "Some features are good, but overall it's average.", 
    "I liked the design but hated the functionality.", 
    "Mixed feelings about this experience.", 
    "It was enjoyable, but I wouldn't go out of my way for it again."
]

labels = [
    # Positive labels
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,

    # Negative labels
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 

    # Neutral/mixed labels (optional: treat as 0 or remove)
    0, 0, 0, 0, 0, 0
]


# Tokenize data
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)

# Custom Dataset
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

dataset = SentimentDataset(inputs, labels)

# Train-validation split using stratification
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# Tokenize train and validation data
train_inputs = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
val_inputs = tokenizer(val_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)

# Create train and validation datasets
train_dataset = SentimentDataset(train_inputs, train_labels)
val_dataset = SentimentDataset(val_inputs, val_labels)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

# Optimizer and Loss
optimizer = AdamW(model.parameters(), lr=5e-5)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")



# Evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

print(classification_report(true_labels, predictions, target_names=["Negative", "Positive"], zero_division=0))




#Faire des prédictions sur de nouveaux textes

# Nouveau texte
new_texts = ["The product is amazing!", "I didn't like the service."]
inputs = tokenizer(new_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
inputs = {key: val.to(device) for key, val in inputs.items()}

# Prédiction
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1)
    sentiments = ["Negative" if pred == 0 else "Positive" for pred in preds]

print(sentiments)
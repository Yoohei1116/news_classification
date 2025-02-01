import torch
import pandas as pd
import numpy as np
import time 
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

MODEL_NAME = "cl-tohoku/bert-base-japanese"  # 日本語対応のBERT

class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df["sentence"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # BERT の入力フォーマットに変換
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

def train_model(model, train_loader, optimizer, num_epochs=5):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()

    start_time = time.time()     
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    end_time = time.time()  
    elapsed_time = end_time - start_time
    print(f"学習時間: {elapsed_time:.2f} 秒")

def evaluate_model(model, test_loader):
    model.eval()
    predictions, true_labels = [], []

    start_time = time.time()  
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"推論時間: {elapsed_time:.2f} 秒")

    f1 = f1_score(true_labels, predictions, average="weighted")
    print(f"F1スコア: {f1:.4f}")

if __name__ == "__main__":
    data = pd.read_csv("news_data.csv")
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=1)
    num_labels = len(data["label"].unique())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    train_dataset = NewsDataset(train_df, tokenizer)
    test_dataset = NewsDataset(test_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    train_model(model, train_loader, optimizer, num_epochs=5) 
    evaluate_model(model, test_loader)  

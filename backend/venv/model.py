import os

# Create the directory if it doesn't exist
os.makedirs("bert_model_with_tokenizer", exist_ok=True)

import os
os.environ["WANDB_DISABLED"] = "true"

import torch
import pandas as pd
import torch.nn as nn
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load CSV dataset
df = pd.read_csv("C:\\Users\\91638\\Downloads\\dataset.csv")

# Combine answer_key and student_answer into a single input
df["input_text"] = df["answer_key"] + " [SEP] " + df["student_answer"]

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Custom Dataset Class
class MarksDataset(Dataset):
    def __init__(self, texts, total_marks, targets):
        self.encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        self.total_marks = torch.tensor(total_marks, dtype=torch.float).unsqueeze(1)
        self.targets = torch.tensor(targets, dtype=torch.float)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["total_marks"] = self.total_marks[idx]
        item["labels"] = self.targets[idx]
        return item

    def __len__(self):
        return len(self.targets)

# Define the model
class BertRegressor(nn.Module):
    def __init__(self):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.total_mark_embed = nn.Linear(1, 32)
        self.regressor = nn.Linear(self.bert.config.hidden_size + 32, 1)

    def forward(self, input_ids, attention_mask, total_marks, labels=None):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        total_marks = total_marks.to(device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        total_embed = self.total_mark_embed(total_marks)
        combined = torch.cat((pooled_output, total_embed), dim=1)
        score = self.regressor(combined).squeeze(1)

        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss_fn = nn.MSELoss()
            loss = loss_fn(score, labels)
        return {"loss": loss, "logits": score}

# Prepare dataset
dataset = MarksDataset(
    df["input_text"].tolist(),
    df["total_marks"].tolist(),
    df["marks_scored"].tolist()
)

# Initialize and move model to device
model = BertRegressor().to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=6,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no"
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train the model
trainer.train()

# Save the model
torch.save(model.state_dict(), "bert_model_with_tokenizer/bert_regressor.pth")

# Save tokenizer
tokenizer.save_pretrained("bert_model_with_tokenizer/saved_tokenizer")

# Save base BERT model (this saves config.json and pytorch_model.bin properly)
model.bert.save_pretrained("bert_model_with_tokenizer/base_bert")
print('model saved')

# Prediction function
def predict_marks(answer_key, student_answer, total_marks):
    model.eval()
    input_text = answer_key + " [SEP] " + student_answer
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    total_tensor = torch.tensor([[total_marks]], dtype=torch.float).to(device)

    with torch.no_grad():
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            total_marks=total_tensor
        )
    return output["logits"].item()

# Sample prediction
answer_key = "Ohm's Law states that the current through a conductor is directly proportional to the voltage across it, provided temperature is constant."
student_answer = "Ohm's Law says current increases with voltage if temperature stays same."
total_marks = 2

predicted = predict_marks(answer_key, student_answer, total_marks)
print(f"Predicted Marks: {round(predicted, 2)}")

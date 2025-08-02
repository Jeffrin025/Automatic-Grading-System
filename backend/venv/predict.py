import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert_model_with_tokenizer/saved_tokenizer")

# Define the same model class structure
class BertRegressor(nn.Module):
    def __init__(self):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained("bert_model_with_tokenizer/base_bert")
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

# Initialize model and load weights
model = BertRegressor().to(device)
model.load_state_dict(torch.load("bert_model_with_tokenizer/bert_regressor.pth"))
model.eval()
def predict_marks(answer_key, student_answer, total_marks):
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
answer_key = "The diqestive system breaks down food into nutrients that the body can absorb and use for enerqy and qrowth."
student_answer = "The diqestive System helps break down food into Small parts So the body cAn absorb nutrients."
total_marks = 2

predicted = predict_marks(answer_key, student_answer, total_marks)
print(f"Predicted Marks: {round(predicted, 2)}")

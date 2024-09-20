from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_json("Project1_Data.json")
df = df.rename(columns={'label': 'labels'})
df['labels'] = df['labels'].astype(int)
df.drop(['id', 'title'], axis=1, inplace = True )
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

# Model id to load the tokenizer
model_id = "vinai/bartpho-syllable"
# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length= 256)

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.question = dataframe['question']
        self.text = dataframe['text']
        self.labels = dataframe['labels']
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        question = str(self.question[index])
        text = str(self.text[index])
        label = self.labels[index]

        # Tokenize the text
        inputs = self.tokenizer(
            question,
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation='only_second',
            return_tensors='pt'
        )

        # Return a dictionary containing input_ids, attention_mask, and labels
        return {
            'input_ids': inputs['input_ids'].squeeze(0),  # Remove extra batch dimension
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = TextDataset(dataframe=train_df, tokenizer=tokenizer, max_length=256)
test_dataset = TextDataset(dataframe=test_df, tokenizer=tokenizer, max_length=256)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

import torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoModel

class BartPhoWithLinear(nn.Module):
    def __init__(self, num_labels=2, dropout=0.1):
        super(BartPhoWithLinear, self).__init__()
        self.bartpho = AutoModel.from_pretrained("vinai/bartpho-syllable")
        
        # Thêm các lớp Linear
        self.fc1 = nn.Linear(1024, 512) 
        self.fc2 = nn.Linear(512, num_labels)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None):
        # Forward qua MBart model
        outputs = self.bartpho(input_ids=input_ids, 
                             attention_mask=attention_mask, 
                             decoder_input_ids=decoder_input_ids, 
                             decoder_attention_mask=decoder_attention_mask)

        # Lấy hidden state cuối cùng từ decoder
        decoder_outputs = outputs.last_hidden_state

        # Lấy vector CLS từ decoder output (tại vị trí 0)
        cls_vector = decoder_outputs[:, 0, :]
        
        # Forward qua các lớp Linear
        x = self.dropout(cls_vector)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Khởi tạo mô hình
model = BartPhoWithLinear(num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

from tqdm import tqdm
import os

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: Epoch {epoch}, Loss: {loss}")
    return model, optimizer, epoch, loss

checkpoint_path = './checkpoints/checkpoint_epoch_5.pt'
model, optimizer, epoch, loss = load_checkpoint(checkpoint_path, model, optimizer)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Đặt mô hình về chế độ đánh giá
model.eval()

# Biến lưu trữ các kết quả
all_preds = []
all_labels = []
eval_loss = 0.0

# Không tính toán gradient trong quá trình đánh giá
with torch.no_grad():
    test_loader_tqdm = tqdm(test_loader, desc="Evaluating")
    
    for batch in test_loader_tqdm:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward qua mô hình
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Tính loss
        loss = criterion(outputs, labels)
        eval_loss += loss.item()
        
        # Dự đoán lớp
        preds = torch.argmax(outputs, dim=1)
        
        # Lưu trữ kết quả
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Tính loss trung bình trên tập kiểm tra
eval_loss /= len(test_loader)

# Tính độ chính xác
accuracy = accuracy_score(all_labels, all_preds)

# In báo cáo phân loại và ma trận nhầm lẫn
print(f"Test Loss: {eval_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

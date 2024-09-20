from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_json("Project1_Data.json")
df = df.rename(columns={'label': 'labels'})
df['labels'] = df['labels'].astype(str)
df.drop(['id', 'title'], axis=1, inplace = True )
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
train_dataset = train_dataset.remove_columns(['__index_level_0__'])
test_dataset = test_dataset.remove_columns(['__index_level_0__'])

label_names = train_dataset.unique('labels')
# Tạo ánh xạ từ nhãn string sang số nguyên
label_to_id = {label: i for i, label in enumerate(label_names)}

# Áp dụng ánh xạ cho cột labels
def label_to_int(example):
    example['labels'] = label_to_id[example['labels']]
    return example

train_dataset = train_dataset.map(label_to_int)
test_dataset = test_dataset.map(label_to_int)

from datasets import ClassLabel

# Tạo ClassLabel từ các nhãn đã xác định
class_label = ClassLabel(names=label_names)

# Chuyển đổi cột labels sang ClassLabel
train_dataset = train_dataset.cast_column("labels", class_label)
test_dataset = test_dataset.cast_column("labels", class_label)

import torch
from transformers import AutoTokenizer

INPUT_MAX_LEN = 512
# Model id to load the tokenizer
model_id = "vinai/bartpho-syllable"
# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length= INPUT_MAX_LEN)
# Tokenize helper function
def tokenize(batch):
    encoding = tokenizer(
        batch['question'],
        batch['text'],
        add_special_tokens=True,
        padding='max_length',  # Ensure all sequences are of the same length
        truncation='only_second',  # Truncate the second sequence if it's too long
        max_length=INPUT_MAX_LEN,  # You can adjust this to your preferred max length
        return_tensors="pt"
    )
    
    # Convert tensors to lists before returning
    return {
        'input_ids': encoding['input_ids'].tolist(),
        'attention_mask': encoding['attention_mask'].tolist(),
    }

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["question", "text"])
test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=["question", "text"])

from transformers import AutoModelForSequenceClassification

# Model id to load the tokenizer
model_id = "vinai/bartpho-syllable"

# Prepare model labels - useful for inference
labels = train_dataset.features["labels"].names
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# Download the model from huggingface.co/models
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
)

import evaluate
import numpy as np

# Metric Id
metric = evaluate.load("f1")

# Metric helper method
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(predictions)
    # Nếu predictions là một tuple, hãy lấy phần tử đầu tiên (logits)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    print(f"Predictions shape before argmax: {predictions.shape}")
    predictions = np.argmax(predictions, axis=1)
    print(f"Predictions shape after argmax: {predictions.shape}")
    print(f"Labels: {labels}")
    
    return metric.compute(predictions=predictions, references=labels, average="weighted")

from transformers import Trainer, TrainingArguments
save_model = "./vinai-bartpho-syllable"

# Define training args
training_args = TrainingArguments(
    output_dir=save_model,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=5,
	torch_compile=True, # optimizations
    optim="adamw_torch_fused", # improved optimizer
    # logging & evaluation strategies
    logging_dir=f"{save_model}/logs",
    logging_strategy="steps",
    logging_steps=500,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
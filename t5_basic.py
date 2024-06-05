from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
from data_preparation_basic import load_and_split_data  
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer 
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSequenceClassification.from_pretrained("t5-base", num_labels=2, ignore_mismatched_sizes=True)

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model.to(device)  # Move the model to the GPU 

# Load and split data 
texts_train, texts_test, labels_train, labels_test = load_and_split_data("FINAL_empathy_dataset.csv")


# Construct classification prompts
train_prompts = ["Does the following text contain empathy?" + text for text in texts_train]
test_prompts = ["Does the following text contain empathy?" + text for text in texts_test]


input_ids = []
attention_masks = []

# Tokenize and preprocess training data
for prompt_text in tqdm(train_prompts):
    encoded_dict = tokenizer.encode_plus(
        prompt_text,
        add_special_tokens=True,
        max_length=256,  
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True
    )
    input_ids.append(encoded_dict["input_ids"])
    attention_masks.append(encoded_dict["attention_mask"])

input_ids = torch.cat(input_ids, dim=0).to(device)
attention_masks = torch.cat(attention_masks, dim=0).to(device)
labels_train = torch.tensor(labels_train).to(device)

# Create a DataLoader for training
train_batch_size = 8  
train_dataset = TensorDataset(input_ids, attention_masks, labels_train)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 3
for epoch in tqdm(range(num_epochs)):
    model.train()
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        input_ids_batch, attention_mask_batch, labels_batch = batch
        input_ids_batch, attention_mask_batch, labels_batch = input_ids_batch.to(device), attention_mask_batch.to(device), labels_batch.to(device)
        outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
        loss = outputs.loss
        
        # Check if loss is a scalar
        if loss.dim() != 0:
            # Reduce the loss to a scalar value
            loss = loss.mean()


        
        loss.backward()
        optimizer.step()

model.save_pretrained('t5')

# Tokenize and preprocess test data
input_ids_test = []
attention_masks_test = []
for prompt_text in tqdm(test_prompts):
    encoded_dict = tokenizer.encode_plus(
        prompt_text,
        add_special_tokens=True,
        max_length=256,  
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True
    )
    input_ids_test.append(encoded_dict["input_ids"])
    attention_masks_test.append(encoded_dict["attention_mask"])

input_ids_test = torch.cat(input_ids_test, dim=0).to(device)
attention_masks_test = torch.cat(attention_masks_test, dim=0).to(device)
labels_test = torch.tensor(labels_test).to(device)

# DataLoader for testing
test_batch_size = 8  
test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# Evaluation on the test set
model.eval()

with torch.no_grad():
    all_predicted_labels = []
    for test_batch in tqdm(test_dataloader):
        input_ids_batch, attention_mask_batch, labels_batch = test_batch
        input_ids_batch, attention_mask_batch, labels_batch = input_ids_batch.to(device), attention_mask_batch.to(device), labels_batch.to(device)
        outputs = model(input_ids_batch, attention_mask=attention_mask_batch)
        logits = outputs.logits
        predicted_labels_batch = torch.argmax(logits, dim=1)
        all_predicted_labels.append(predicted_labels_batch.cpu())

# Concatenate predicted labels from all batches
predicted_labels = torch.cat(all_predicted_labels)

# Calculate metrics
accuracy = accuracy_score(labels_test.cpu(), predicted_labels.cpu())

# Calculate additional metrics
precision, recall, f1_score, _ = precision_recall_fscore_support(labels_test.cpu(), predicted_labels.cpu(), average=None)

# Print the results
print(f"Test Accuracy: {accuracy}")
print(f'Precision for Class 0: {precision[0]:.4f}')
print(f'Precision for Class 1: {precision[1]:.4f}')
print(f'Recall for Class 0: {recall[0]:.4f}')
print(f'Recall for Class 1: {recall[1]:.4f}')
print(f'F1 Score for Class 0: {f1_score[0]:.4f}')
print(f'F1 Score for Class 1: {f1_score[1]:.4f}')

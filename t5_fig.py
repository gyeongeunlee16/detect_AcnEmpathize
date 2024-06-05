from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from data_preparation import load_and_split_data

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained T5-base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSequenceClassification.from_pretrained("t5-base", num_labels=2)

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model.to(device)

# Split data into training and test sets (80:20 split) and load figurative language features
texts_train, texts_test, labels_train, labels_test, idioms_train, idioms_test, metaphors_train, metaphors_test, hyperboles_train, hyperboles_test = load_and_split_data("FINAL_empathy_dataset_figurative.csv")

# Function to append figurative language features to text and construct prompts
def construct_prompts_with_features(texts, idioms, metaphors, hyperboles):
    return ["Given the following text, classify whether it contains empathy or not " + text + f" [IDIOM {idiom}] [METAPHOR {metaphor}] [HYPERBOLE {hyperbole}]" for text, idiom, metaphor, hyperbole in zip(texts, idioms, metaphors, hyperboles)]

# Construct classification prompts with figurative language features
train_prompts = construct_prompts_with_features(texts_train, idioms_train, metaphors_train, hyperboles_train)
test_prompts = construct_prompts_with_features(texts_test, idioms_test, metaphors_test, hyperboles_test)

# tokenize and preprocess data
def tokenize_and_preprocess(texts):
    input_ids = []
    attention_masks = []

    for text in tqdm(texts):
        encoded_dict = tokenizer.encode_plus(                                        
            text,
            add_special_tokens=True,
            max_length=256,  
            padding='max_length',
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    return input_ids, attention_masks

# Tokenize and preprocess training and test data
input_ids_train, attention_masks_train = tokenize_and_preprocess(train_prompts)
labels_train = torch.tensor(labels_train).to(device)
input_ids_test, attention_masks_test = tokenize_and_preprocess(test_prompts)
labels_test = torch.tensor(labels_test).to(device)

# DataLoaders for training and testing
train_batch_size = 8  
train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

test_batch_size = 8  
test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)

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
        
#model.save_pretrained('t5_fig')

# Evaluation on the test set
model.eval()
all_predicted_labels = []
with torch.no_grad():
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
precision, recall, f1_score, _ = precision_recall_fscore_support(labels_test.cpu(), predicted_labels.cpu(), average=None)

# Print the results
print(f"Test Accuracy: {accuracy}")
print(f'Precision for Class 0: {precision[0]:.4f}')
print(f'Precision for Class 1: {precision[1]:.4f}')
print(f'Recall for Class 0: {recall[0]:.4f}')
print(f'Recall for Class 1: {recall[1]:.4f}') 
print(f'F1 Score for Class 0: {f1_score[0]:.4f}')
print(f'F1 Score for Class 1: {f1_score[1]:.4f}')

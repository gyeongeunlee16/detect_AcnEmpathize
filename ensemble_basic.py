from transformers import AutoTokenizer
from data_preparation import load_and_split_data
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


_, texts_test, _, labels_test = load_and_split_data("FINAL_empathy_dataset.csv", test_size=0.2)

# Load tokenizers
tokenizer_roberta_twitter = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
tokenizer_roberta_large_mnli = AutoTokenizer.from_pretrained("roberta-large-mnli")
tokenizer_t5 = AutoTokenizer.from_pretrained("t5-base")


# Tokenize the test data for each model
def tokenize_for_model(tokenizer, texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=256)

tokenized_data_roberta_twitter = tokenize_for_model(tokenizer_roberta_twitter, texts_test)
tokenized_data_roberta_large_mnli = tokenize_for_model(tokenizer_roberta_large_mnli, texts_test)
tokenized_data_t5 = tokenize_for_model(tokenizer_t5, texts_test)


def create_dataloader(tokenized_data, labels):
    dataset = TensorDataset(tokenized_data['input_ids'], tokenized_data['attention_mask'], torch.tensor(labels))
    return DataLoader(dataset, batch_size=32)


# Create DataLoaders for each tokenized dataset
dataloader_roberta_twitter = create_dataloader(tokenized_data_roberta_twitter, labels_test)
dataloader_roberta_large_mnli = create_dataloader(tokenized_data_roberta_large_mnli, labels_test)
dataloader_t5 = create_dataloader(tokenized_data_t5, labels_test)


# Load the fine-tuned models from google drive
model_t5 = AutoModelForSequenceClassification.from_pretrained('/content/t5_pretrained').to(device)
model_roberta_large_mnli = AutoModelForSequenceClassification.from_pretrained('/content/roberta_large_mnli_pretrained').to(device)
model_roberta_twitter = AutoModelForSequenceClassification.from_pretrained('drive/MyDrive/roberta_twitter_updated_pretrained').to(device)


def get_model_predictions(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, attention_mask = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.append(logits)
    return torch.cat(predictions, dim=0)


# Get predictions from each model
predictions_roberta_twitter = get_model_predictions(model_roberta_twitter, dataloader_roberta_twitter)
predictions_roberta_large_mnli = get_model_predictions(model_roberta_large_mnli, dataloader_roberta_large_mnli)
predictions_t5 = get_model_predictions(model_t5, dataloader_t5)


prob_roberta_twitter = F.softmax(predictions_roberta_twitter, dim=1)
prob_roberta_large_mnli = F.softmax(predictions_roberta_large_mnli, dim=1)
prob_t5 = F.softmax(predictions_t5, dim=1)

# Calculate confidence (max probability) for each model's predictions
confidence_roberta_twitter = torch.max(prob_roberta_twitter, dim=1).values
confidence_roberta_large_mnli = torch.max(prob_roberta_large_mnli, dim=1).values
confidence_t5 = torch.max(prob_t5, dim=1).values

# Weight predictions by their confidence
weighted_pred_roberta_twitter = (prob_roberta_twitter.T * confidence_roberta_twitter).T
weighted_pred_roberta_large_mnli = (prob_roberta_large_mnli.T * confidence_roberta_large_mnli).T
weighted_pred_t5 = (prob_t5.T * confidence_t5).T

# Sum the weighted predictions
summed_weighted_predictions = weighted_pred_roberta_twitter + weighted_pred_roberta_large_mnli + weighted_pred_t5

# Determine final predictions (class with highest summed weighted prediction)
final_predictions = torch.argmax(summed_weighted_predictions, dim=1)

# Calculate metrics
accuracy = accuracy_score(labels_test, final_predictions)
precision, recall, f1_score, _ = precision_recall_fscore_support(labels_test, final_predictions, average=None)

# Print the results
print(f"Weighted Confidence Ensemble Test Accuracy: {accuracy}")
for i, (prec, rec, f1) in enumerate(zip(precision, recall, f1_score)):
    print(f'Class {i} - Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}')



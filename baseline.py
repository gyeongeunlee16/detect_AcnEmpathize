import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv("Fig_LIWC.csv")

liwc_columns = ['Analytic', 'Linguistic', 'function', 'insight', 'feeling']


# Extract selected LIWC columns
texts = df['text']
liwc_features = df[['Analytic', 'Linguistic', 'function', 'insight', 'feeling']]
y = df['combined_empathy']  # target variable

# Split the dataset into training and testing sets
texts_train, texts_test, liwc_features_train, liwc_features_test, y_train, y_test = train_test_split(
                    texts, liwc_features, y, test_size=0.2, random_state=42)

# truncate texts to a maximum of 256 characters
def truncate_texts(texts):
        return [text[:256] for text in texts]

# combine LIWC features with modified texts
def combine_features(liwc_features, modified_texts, liwc_columns):
    combined_features = []
    for row, text in zip(liwc_features.iterrows(), modified_texts):
        selected_liwc = ' '.join([str(row[1][column]) for column in liwc_columns])
        combined_features.append(text + ' ' + selected_liwc)
    return combined_features

# truncate texts to a maximum of 256 characters
modified_texts_train_truncated = truncate_texts(texts_train)
modified_texts_test_truncated = truncate_texts(texts_test)

# Select LIWC features
liwc_columns = ['Analytic', 'Linguistic', 'function', 'insight', 'feeling']


# Combine LIWC features with truncated texts
X_train_combined = combine_features(liwc_features_train, modified_texts_train_truncated, liwc_columns)
X_test_combined = combine_features(liwc_features_test, modified_texts_test_truncated, liwc_columns)

# Vectorize the combined features
vectorizer = CountVectorizer(max_features=256)
X_train_vectorized = vectorizer.fit_transform(X_train_combined)
X_test_vectorized = vectorizer.transform(X_test_combined)

# Train and evaluate classifiers
nb_classifier = MultinomialNB()
svm_classifier = SVC(kernel='linear')
lr_classifier = LogisticRegression(max_iter=1000)

nb_classifier.fit(X_train_vectorized, y_train)
svm_classifier.fit(X_train_vectorized, y_train)
lr_classifier.fit(X_train_vectorized, y_train)

nb_pred_truncated = nb_classifier.predict(X_test_vectorized)
svm_pred_truncated = svm_classifier.predict(X_test_vectorized)
lr_pred_truncated = lr_classifier.predict(X_test_vectorized)

nb_accuracy_truncated = accuracy_score(y_test, nb_pred_truncated)
nb_precision_truncated = precision_score(y_test, nb_pred_truncated, average=None)
nb_recall_truncated = recall_score(y_test, nb_pred_truncated, average=None)
nb_f1_truncated = f1_score(y_test, nb_pred_truncated, average=None)

svm_accuracy_truncated = accuracy_score(y_test, svm_pred_truncated)
svm_precision_truncated = precision_score(y_test, svm_pred_truncated, average=None)
svm_recall_truncated = recall_score(y_test, svm_pred_truncated, average=None)
svm_f1_truncated = f1_score(y_test, svm_pred_truncated, average=None)

lr_accuracy_truncated = accuracy_score(y_test, lr_pred_truncated)
lr_precision_truncated = precision_score(y_test, lr_pred_truncated, average=None)
lr_recall_truncated = recall_score(y_test, lr_pred_truncated, average=None)
lr_f1_truncated = f1_score(y_test, lr_pred_truncated, average=None)

print("Naive Bayes Performance:")
print("Accuracy:", nb_accuracy_truncated)
print("Precision:", nb_precision_truncated)
print("Recall:", nb_recall_truncated)
print("F1 Score:", nb_f1_truncated)

print("\nSVM Performance:")
print("Accuracy:", svm_accuracy_truncated)
print("Precision:", svm_precision_truncated)
print("Recall:", svm_recall_truncated)
print("F1 Score:", svm_f1_truncated)

print("\nLogistic Regression Performance:")
print("Accuracy:", lr_accuracy_truncated)
print("Precision:", lr_precision_truncated)
print("Recall:", lr_recall_truncated)
print("F1 Score:", lr_f1_truncated)


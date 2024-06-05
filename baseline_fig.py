import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
file_path = 'Fig_LIWC.csv'
df = pd.read_csv(file_path)

texts = df['text']
idioms = df['Idiom']
metaphors = df['Metaphor']
hyperboles = df['Hyperbole']
liwc_features = df[['Analytic', 'Linguistic', 'function', 'insight', 'feeling']]
y = df['combined_empathy']

# Split the dataset into training and testing sets
texts_train, texts_test, idioms_train, idioms_test, metaphors_train, metaphors_test, hyperboles_train, hyperboles_test, liwc_features_train, liwc_features_test, y_train, y_test = train_test_split(
            texts, idioms, metaphors, hyperboles, liwc_features, y, test_size=0.2, random_state=42)

# append figurative language features to text
def append_features(texts, idioms, metaphors, hyperboles):
        return [f"{text} [IDIOM {idiom}] [METAPHOR {metaphor}] [HYPERBOLE {hyperbole}]" for text, idiom, metaphor, hyperbole in zip(texts, idioms, metaphors, hyperboles)]


# combine LIWC features with modified texts
def combine_features(liwc_features, modified_texts, liwc_columns):
    combined_features = []
    for row, text in zip(liwc_features.iterrows(), modified_texts):
        selected_liwc = ' '.join([str(row[1][column]) if isinstance(row[1][column], (int, float)) else str(row[1][column]) for column in liwc_columns])
        combined_features.append(selected_liwc + ' ' + text)
    return combined_features


# Append figurative language features to the texts
modified_texts_train = append_features(texts_train, idioms_train, metaphors_train, hyperboles_train)
modified_texts_test = append_features(texts_test, idioms_test, metaphors_test, hyperboles_test)


# truncate texts to a maximum of 256 characters
def truncate_texts(texts, max_chars=256):
    truncated_texts = []
    for text in texts:
        truncated_texts.append(text[:max_chars])
    return truncated_texts

# truncate texts to a maximum of 256 characters
modified_texts_train_truncated = truncate_texts(modified_texts_train)
modified_texts_test_truncated = truncate_texts(modified_texts_test)

liwc_columns = ['Analytic', 'Linguistic', 'function', 'insight', 'feeling']



# Combine LIWC features with truncated texts
X_train_combined_truncated = combine_features(liwc_features_train, modified_texts_train_truncated, liwc_columns)
X_test_combined_truncated = combine_features(liwc_features_test, modified_texts_test_truncated, liwc_columns)

# Vectorize the combined features
vectorizer = CountVectorizer(max_features=256)
X_train_vectorized_truncated = vectorizer.fit_transform(X_train_combined_truncated)
X_test_vectorized_truncated = vectorizer.transform(X_test_combined_truncated)

# Train and evaluate classifiers
nb_classifier = MultinomialNB()
svm_classifier = SVC(kernel='linear')
lr_classifier = LogisticRegression(max_iter=1000)

nb_classifier.fit(X_train_vectorized_truncated, y_train)
svm_classifier.fit(X_train_vectorized_truncated, y_train)
lr_classifier.fit(X_train_vectorized_truncated, y_train)

nb_pred_truncated = nb_classifier.predict(X_test_vectorized_truncated)
svm_pred_truncated = svm_classifier.predict(X_test_vectorized_truncated)
lr_pred_truncated = lr_classifier.predict(X_test_vectorized_truncated)

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


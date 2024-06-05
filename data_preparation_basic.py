# train test split for basic models

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath, test_size=0.2, random_state=42):
    # Load the dataset
    data = pd.read_csv(filepath)
    texts = data['text'].tolist()
    labels = data['combined_empathy'].tolist()

    # Split the data
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=test_size, random_state=random_state)
                                
    return texts_train, texts_test, labels_train, labels_test

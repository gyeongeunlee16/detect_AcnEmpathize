# file to split train & test sets with figurative language features

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath, test_size=0.2, random_state=42):
    # Load the dataset
    data = pd.read_csv(filepath)
    texts = data['text'].tolist()  
    labels = data['combined_empathy'].tolist()  
    idioms = data['Idiom'].tolist()  
    metaphors = data['Metaphor'].tolist()  
    hyperboles = data['Hyperbole'].tolist()  

    # Split the data
    return train_test_split(texts, labels, idioms, metaphors, hyperboles, test_size=test_size, random_state=random_state)


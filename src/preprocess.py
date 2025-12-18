import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text:str) -> str:
    text = text.lower()
    text  = re.sub(r'\d+', 'NUM', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text 

def preprocess_data(csv_path:str):
    df = pd.read_csv(csv_path)
    df['clean_text']  =df['email_text'].apply(clean_text)
    X = df['clean_text']
    y = df['priority']
    return X,y
def vectorizer():
    return TfidfVectorizer(max_features=5000, stop_words = 'english')    
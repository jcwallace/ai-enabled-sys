## Imports
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Read Data from CSV
amazMusical_df = pd.read_csv('archive/Musical_instruments_reviews.csv')

# Retrieve Summary Data
summary = amazMusical_df['summary']
summary = ' '.join(str(s) for s in summary)

# Tokenize Summary
tokens = word_tokenize(summary)

# Stem the Tokens
st = LancasterStemmer()
stems = [st.stem(tok) for tok in tokens]

# Lemmatize the stemmed tokens
lem = WordNetLemmatizer()
lemms = [lem.lemmatize(stem) for stem in stems] 

# Reform all data into dataframe
data = {'Tokens': tokens,
        'Stems': stems,
        'Lemmenize': lemms}
processed_df = pd.DataFrame(data,columns=["Tokens","Stems","Lemmenize"])

print( processed_df )
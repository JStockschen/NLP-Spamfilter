import sys
import pandas as pd
import nltk
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Functie voor Part of Speech tagging voor betere lemmatizatoin
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # Default is een zelfstandig naamwoord

def program():
    # Lees CSV bestand
    df = pd.read_csv('emails.csv')

    # Initializeer WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lijst voor cleaned texts
    cleaned_texts = []

    # Ga over elke rij in dataframe
    for index, row in df.iterrows():
        text = row['text']  # selecteer kolom met tekst
        tokens = word_tokenize(text)  # Maak tokens van de tekst

        # Filter non-alfabetische karakters met regex
        alphabetic_tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
        alphabetic_tokens = [token for token in alphabetic_tokens if token]  # Verwijder lege tokens (spaties)

        # Lemmetizer toepassen om vergelijkbare tokens samen te voegen
        lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in alphabetic_tokens]

        # Verander tokens terug naar string
        cleaned_text = ' '.join(lemmatized_tokens)
        cleaned_texts.append(cleaned_text)

    # Kies kolom
    y = df['text']

    # Vectorize de strings
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_texts)

    # Split data in training en test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)

    # evalueer model
    score = clf.score(X_test, y_test)

    print(f"Model accuracy: {score}")

program()
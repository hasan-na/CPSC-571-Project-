import os
import string
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import re
import tensorflow as tf
import pandas as pd #type: ignore
import numpy as np #type: ignore
from gensim.models import Word2Vec
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Bidirectional, LSTM, Dense # type: ignore

def main():
   
    # Load data
    train_data = pd.read_csv(os.path.join('data', 'train.csv'))
    texts = train_data['comment_text']
    labels = train_data[train_data.columns[2:]] 

    # Tokenize the text
    tokenized_texts = [tokenize(text) for text in texts] 

    # Train Word2Vec model
    w2v_model = Word2Vec(tokenized_texts, min_count=1)
    
    # Convert words to their corresponding word vectors
    vectorized_texts = [[w2v_model.wv[word] for word in text] for text in tokenized_texts]
    
    # Pad the sequences so they're all the same length
    max_sequence_length = 100  
    vectorized_texts = pad_sequences(vectorized_texts, maxlen=max_sequence_length)
    vectorized_texts_train, vectorized_texts_val, labels_train, labels_val = train_test_split(vectorized_texts, labels, test_size=0.2, random_state=42)

    # Create a neural network
    model = Sequential()
    model.add(Bidirectional(LSTM(32, activation = 'tanh')))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train the model
    model.fit(vectorized_texts_train, labels_train, epochs=10, batch_size=32)

    loss, accuracy = model.evaluate(vectorized_texts_val, labels_val)
    # Calculate precision, recall, and F1 score for each class
    print('Test Loss:', loss)
    print('Test Accuracy:', accuracy)

    # Make predictions on the test data
    test_probabilities = model.predict(vectorized_texts_val)
    test_predictions = (test_probabilities > 0.35).astype(int)

    # Calculate precision, recall, and F1 score for each class
    precision = precision_score(labels_val, test_predictions, average=None)
    recall = recall_score(labels_val, test_predictions, average=None)
    f1 = f1_score(labels_val, test_predictions, average=None)

    # Print the evaluation results
    labels_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for i, label in enumerate(labels_list):
        print(f'Class: {label}')
        print(f'  Precision: {precision[i]}')
        print(f'  Recall: {recall[i]}')
        print(f'  F1 Score: {f1[i]}')

def tokenize(text):
    # Convert the text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Tokenize the text
    tokens = text.split()
    # Remove words that are less than 2 characters long
    tokens = [token for token in tokens if len(token) > 2]
    return tokens
    

if __name__ == '__main__':
    main()

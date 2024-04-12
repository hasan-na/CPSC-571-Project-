import os
import string
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import re
import tensorflow as tf
import pandas as pd #type: ignore
import numpy as np #type: ignore
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Bidirectional, LSTM, Dense # type: ignore

def main():
   
    # Load data
    train_data = pd.read_csv(os.path.join('data', 'train.csv'))
    test_data = pd.read_csv(os.path.join('data', 'test.csv'))
    test_data_labels = pd.read_csv(os.path.join('data', 'test_labels.csv'))
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
    model.fit(vectorized_texts, labels, epochs=1, batch_size=32, verbose=1)

    # Tokenize the test text
    test_texts = test_data['comment_text']
    tokenized_test_texts = [tokenize(text) for text in test_texts]
    # Convert words to their corresponding word vectors
    vectorized_test_texts = [[w2v_model.wv[word] for word in text if word in w2v_model.wv] for text in tokenized_test_texts]
    # Pad the sequences so they're all the same length
    vectorized_test_texts = pad_sequences(vectorized_test_texts, maxlen=max_sequence_length)
    # Get the test labels
    test_labels = test_data_labels[test_data_labels.columns[1:]]
    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(vectorized_test_texts, test_labels)
    print('Test Loss:', loss)
    print('Test Accuracy:', accuracy)

    # Make predictions
    prompt = 'You are a terrible person.'
    prediction = predict_toxicity(model, w2v_model, prompt)

    if prediction:
        categories = ', '.join(prediction)
        print(f'The prompt ({prompt}) contains {categories} which has been filtered out.')
    else:
        print('The prompt is not toxic.')
   
    
def predict_toxicity(model, w2v_model, text):
    # Tokenize the text
    tokenized_text = tokenize(text)
    # Convert words to their corresponding word vectors
    vectorized_text = [w2v_model.wv[word] for word in tokenized_text if word in w2v_model.wv]
    # Pad the sequence so it's the same length as the training data
    max_sequence_length = 100
    vectorized_text = pad_sequences([vectorized_text], maxlen=max_sequence_length)
    # Make a prediction
    probabilities = model.predict(vectorized_text)[0]
    # Make labels
    labels_list = ['toxicity', 'severe_toxicity', 'obscene language', 'threats', 'insults', 'identity_hate']
    # Convert the probabilities to category labels
    categories = [labels_list[i] for i, p in enumerate(probabilities) if p > 0.15]
    
    return categories

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

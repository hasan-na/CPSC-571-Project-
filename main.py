import tensorflow as tf
import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Bidirectional, LSTM, Dense # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore

def main():

    #download the punkt package from nltk
    #nltk.download('punkt')

    # Load data
    data = pd.read_csv(os.path.join('data', 'train.csv'))
    texts = data['comment_text']
    labels = data[data.columns[2:]]
    
    # Tokenize the text
    tokenized_texts = [word_tokenize(text) for text in texts]
    
    # Train Word2Vec model
    w2v_model = Word2Vec(tokenized_texts, min_count=1)
    
    # Convert words to their corresponding word vectors
    vectorized_texts = [[w2v_model.wv[word] for word in text] for text in tokenized_texts]
    
    # Pad the sequences so they're all the same length
    max_sequence_length = 100  
    vectorized_texts = pad_sequences(vectorized_texts, maxlen=max_sequence_length)
    
    # Split the data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(vectorized_texts, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

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
    history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_val, y_val), verbose=2)
    
    # Make predictions
    predictions = predict_toxicity(model, w2v_model, 'You are a terrible person')
    print(predictions)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
    #Can add more metrics here

    # # Save the model
    # model.save(os.path.join('models', 'model.h5'))
    # # Save the Word2Vec model
    # w2v_model.save(os.path.join('models', 'w2v_model.h5'))

def predict_toxicity(model, w2v_model, text):
    # Tokenize the text
    tokenized_text = word_tokenize(text)
    # Convert words to their corresponding word vectors
    vectorized_text = [w2v_model.wv[word] for word in tokenized_text]
    # Pad the sequence so it's the same length as the training data
    max_sequence_length = 100
    vectorized_text = pad_sequences([vectorized_text], maxlen=max_sequence_length)
    # Make a prediction
    # Use the model to predict the categories
    probabilities = model.predict(vectorized_text)[0]
    
    # Convert the probabilities to category labels
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']  
    labels = [categories[i] for i, p in enumerate(probabilities) if p > 0.5]
    
    return labels

if __name__ == '__main__':
    main()

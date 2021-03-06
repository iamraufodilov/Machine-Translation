#load libraries
import pandas as pd
import numpy as np
import string
from string import digits
import matplotlib.pyplot as plt
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
import itertools


# load dataset
data_path = 'G:/rauf/STEPBYSTEP/Data/translation_ds/spa.txt'
lines= pd.read_table(data_path,  names =['source', 'target', 'comments'])
#_>print(lines.sample(6))


# clean the data

# convert source and target text to Lowercase 
lines.source=lines.source.apply(lambda x: x.lower())
lines.target=lines.target.apply(lambda x: x.lower())

# Remove quotes from source and target text
lines.source=lines.source.apply(lambda x: re.sub("'", '', x))
lines.target=lines.target.apply(lambda x: re.sub("'", '', x))

# create a set of all special characters
special_characters= set(string.punctuation)

# Remove all the special characters
lines.source = lines.source.apply(lambda x: ''.join(char1 for char1 in x if char1 not in special_characters))
lines.target = lines.target.apply(lambda x: ''.join(char1 for char1 in x if char1 not in special_characters))

# Remove digits from source and target sentences
num_digits= str.maketrans('','', digits)
lines.source=lines.source.apply(lambda x: x.translate(num_digits))
lines.target= lines.target.apply(lambda x: x.translate(num_digits))

# Remove extra spaces
lines.source=lines.source.apply(lambda x: x.strip())
lines.target=lines.target.apply(lambda x: x.strip())
lines.source=lines.source.apply(lambda x: re.sub(" +", " ", x))
lines.target=lines.target.apply(lambda x: re.sub(" +", " ", x))


# Add start and end tokens to target sequences
lines.target = lines.target.apply(lambda x : 'START_ '+ x + ' _END')
#_>print(lines.sample(6))


# Find all the source and target unique words and sort them
# Vocabulary of Source language
all_source_words=set()
for source in lines.source:
    for word in source.split():
        if word not in all_source_words:
            all_source_words.add(word)
# Vocabulary of Target 
all_target_words=set()
for target in lines.target:
    for word in target.split():
        if word not in all_target_words:
            all_target_words.add(word)

# sort all unique source and target words
source_words= sorted(list(all_source_words))
target_words=sorted(list(all_target_words))


#Find maximum sentence length in  the source and target data
source_length_list=[]
for l in lines.source:
    source_length_list.append(len(l.split(' ')))
max_source_length= max(source_length_list)
#_>print(" Max length of the source sentence",max_source_length)
target_length_list=[]
for l in lines.target:
    target_length_list.append(len(l.split(' ')))
max_target_length= max(target_length_list)
#_>print(" Max length of the target sentence",max_target_length)


# creating a word to index(word2idx) for source and target
source_word2idx= dict([(word, i+1) for i,word in enumerate(source_words)])
target_word2idx=dict([(word, i+1) for i, word in enumerate(target_words)])
#_>print(dict(itertools.islice(source_word2idx.items(), 5))) # lets look five examples

# creating a word to index(word2idx) for source and target
source_word2idx= dict([(word, i+1) for i,word in enumerate(source_words)])
target_word2idx=dict([(word, i+1) for i, word in enumerate(target_words)])
#_>print(dict(itertools.islice(source_word2idx.items(), 5))) # lets look five examples


#Shuffle the data
lines = shuffle(lines)


# Train - Test Split
X, y = lines.source, lines.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
#_>print(X_train.shape, X_test.shape)


# Create data for training the encoder-decoder model.

# Input tokens for encoder
num_encoder_tokens=len(source_words)
# Input tokens for decoder zero padded
num_decoder_tokens=len(target_words) +1

def generate_batch(X = X_train, y = y_train, batch_size = 128):
    ''' Generate a batch of data '''
    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = np.zeros((batch_size, max_source_length),dtype='float32')
            decoder_input_data = np.zeros((batch_size, max_target_length),dtype='float32')
            decoder_target_data = np.zeros((batch_size, max_target_length, num_decoder_tokens),dtype='float32')
            for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_text.split()):
                    encoder_input_data[i, t] = source_word2idx[word] 
                for t, word in enumerate(target_text.split()):
                    if t<len(target_text.split())-1:
                        decoder_input_data[i, t] = target_word2idx[word] # decoder input seq
                    if t>0:
                        # decoder target sequence (one hot encoded)
                        # does not include the START_ token
                        # Offset by one timestep
                        #print(word)
                        decoder_target_data[i, t - 1, target_word2idx[word]] = 1.
                    
            yield([encoder_input_data, decoder_input_data], decoder_target_data)


# Build the sequence to sequence model


#Setup basic parameters
train_samples = len(X_train)
val_samples = len(X_test)
batch_size = 128
epochs = 50
latent_dim=256


#Build the Encoder
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
enc_emb =  Embedding(num_encoder_tokens, latent_dim, mask_zero = True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]


#Build the Decoder
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True)
dec_emb = dec_emb_layer(decoder_inputs)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# efine the model
# Define the model that takes encoder and decoder input 
# to output decoder_outputs
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])


#Train the model
train_samples = len(X_train) # Total Training samples
val_samples = len(X_test)    # Total validation or test samples
batch_size = 128
epochs = 100

model.fit_generator(generator = generate_batch(X_train, y_train, batch_size = batch_size),
                    steps_per_epoch = train_samples//batch_size,
                    epochs=epochs,
                    validation_data = generate_batch(X_test, y_test, batch_size = batch_size),
                    validation_steps = val_samples//batch_size)


# save model weights
model.save_weights('nmt_weights_100epochs.h5')


# load model weights
model.load_weights('nmt_weights_100epochs.h5')


# now we create model to translate input sentence to uotput sentence
# Encode the input sequence to get the "Context vectors"
encoder_model = Model(encoder_inputs, encoder_states)
# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_state_input = [decoder_state_input_h, decoder_state_input_c]
# Get the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs)
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_state_input)
decoder_states2 = [state_h2, state_c2]
# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_outputs2)
# Final decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_state_input,
    [decoder_outputs2] + decoder_states2)


# create function to get translated result
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of 
    #target sequence with the start character.
    target_seq[0, 0] = target_word2idx['START_']
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word =target_idx2word[sampled_token_index]
        decoded_sentence += ' '+ sampled_word
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == '_END' or
           len(decoded_sentence) > 50):
            stop_condition = True
            # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        # Update states
        states_value = [h, c]
    return decoded_sentence


# Make predictions on the training dataset
train_gen = generate_batch(X_train, y_train, batch_size = 1)
k=-1

k+=1
(input_seq, actual_output), _ = next(train_gen)
decoded_sentence = decode_sequence(input_seq)
print('Input Source sentence:', X_train[k:k+1].values[0])
print('Actual Target Translation:', y_train[k:k+1].values[0][6:-4])
print('Predicted Target Translation:', decoded_sentence[:-4])


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONCLUSION
'''
in this project we created neural machine translation model
first we load dataset and clean it from spaces, digits, special characters
then we create decoder and encoder pipelines
then we created model with two encoding and decoding pipelines
in the last we created function to translate random input text to desired language
'''
# load libraries
import collections

#import helper
import numpy as np
import pandas as pd
#import project_tests as tests

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


# load dataset
english_path = 'G:/rauf/STEPBYSTEP/Data/translation_ds/english-french/small_vocab_en.txt'
french_path = 'G:/rauf/STEPBYSTEP/Data/translation_ds/english-french/small_vocab_fr.txt'

english_ds=pd.read_csv(english_path,delimiter="\t")
#_>print(english_ds.head)
french_ds=pd.read_csv(french_path,delimiter="\t")
#_>print(french_ds.head)


# get vocabulary
english_words_counter = collections.Counter([word for sentence in english_ds for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_ds for word in sentence.split()])

# lets look our data complexity
print('{} English words.'.format(len([word for sentence in english_ds for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))

print('{} French words.'.format(len([word for sentence in french_ds for word in sentence.split()])))
print('{} unique French words.'.format(len(french_words_counter)))


# tokenize sentences to their respective ids to change string value to integer value
def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer

tests.test_tokenize(tokenize)

# Tokenize Example output
text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']

text_tokenized, text_tokenizer = tokenize(text_sentences)
#_>print(text_tokenizer.word_index)

for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))


# add padding for each sentnece which has less less length than max length sentence, cuz for batching we should make each sentence must have same length
def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    return pad_sequences(x, maxlen=length, padding='post')

tests.test_pad(pad)

# Pad Tokenized output
test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))


# create function for preprocessing pipeline
def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
    preprocess(english_sentences, french_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

#_>print('Data Preprocessed')
#_>print("Max English sentence length:", max_english_sequence_length)
#_>print("Max French sentence length:", max_french_sequence_length)
#_>print("English vocabulary size:", english_vocab_size)
#_>print("French vocabulary size:", french_vocab_size)


# create function to convert model outputs to target string which in our case translated french sentences
def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONCLUSION
'''
this projects steps
load dataset
create vocabulary
tokenize words to ids and usa padding to get every sentence same lenth
then we created function to convert model output to tranlated french strings
in this directory we have several model scripts which can be implemented with several model performance:
1) model RNN
2) model Embedding
3) model Bidirectional
4) model Encoder-Decoder
5) model Costom
'''
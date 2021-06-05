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


# load dataset
#data_path = 'G:/rauf/STEPBYSTEP/Data/translation_ds/spa.txt'
#lines= pd.read_table(data_path,  names =['source', 'target', 'comments'])
#print(lines.sample(6))
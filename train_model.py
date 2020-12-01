# import libraries
import nltk
import json
import pickle
import string
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return lemmatizer.lemmatize(word.lower())
"""
def bag_of_words(tokenizes_sentence, patterns):
    tokenizes_sentence = [stem(w) for w in tokenizes_sentence]
    bag = np.zeros(len(patterns), dtype = np.float32)
    for index, w in enumerate(patterns):
        if w in tokenizes_sentence:
            bag[index] = 1.0
    return bag
"""
# preprocesse input data
patterns = []
tags = []
documents = [] # documents = combination between patterns and tags
ignore_words = string.punctuation
ignore_words = nltk.word_tokenize(ignore_words)

data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    if intent['tag'] not in tags: # add to our tags list
        tags.append(intent['tag'])
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        patterns.extend(w) # w is a array, we can not put a array of arry in words, therefor extend()
        documents.append((w, intent['tag']))
       
patterns = [stem(w) for w in patterns if w not in ignore_words]
patterns = sorted(list(set(patterns)))
tags = sorted(list(set(tags)))


training = []
output_empty = np.zeros(len(tags), dtype = np.float32)
for doc in documents:
    bag = []
    pattern_words = doc[0] # list of tokenized words for the pattern
    pattern_words = [stem(word) for word in pattern_words]
    
    for pattern in patterns:
        bag.append(1) if pattern in pattern_words else bag.append(0) # create our bag of words array with 1, if word match found in current pattern
    output_row = list(output_empty)
    output_row[tags.index(doc[1])] = 1  # output is a '0' for each tag and '1' for current tag (for each pattern)
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

X_train = list(training[:,0])
y_train = list(training[:,1])

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape = (len(X_train[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation = 'softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss ='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(X_train), np.array(y_train), epochs = 200, batch_size = 5, verbose = 1)

pickle.dump(patterns,open('patterns.pkl','wb'))
pickle.dump(tags,open('tags.pkl','wb'))
model.save('chatbot_model.h5', hist)


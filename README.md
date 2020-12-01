# Chatbot_NLP

###### NLTK, TensorFlow, Keras, tkinter

> In this project, I build a chatbot using deep learning techniques. The chatbot will be trained on the dataset which contains categories (intents), pattern and responses. We use a special recurrent neural network (LSTM) to classify which category the user’s message belongs to and then we will give a random response from the list of responses.


### Dataset

intents.json

You can edit the file per your requirment.

### Step 1: train_model.py

- Data Preprocessing

    - Tokenize input sentence 
    - Stemmer each words from tokenized words
    - Lower case
    - Remove punctuations
    
- Input, output dataset

    - Create bag of words 
    - Input is the pattern 
    - Output is the tag input pattern belongs to

- Build the model - 3 layers
    - First layer 128 neurons
    - Second layer 64 neurons 
    - 3rd output layer contains number of neurons equal to number of intents to predict output intent with softmax
- Train the model
    - 200 epoches
- Serialization

### Step 2: user_interface.py

- Tkinter library comes in python

### Inspired by: 

Chat Bot With PyTorch - NLP And Deep Learning - Python Tutorial https://www.youtube.com/watch?v=8qwowmiXANQ

Contextual Chatbots with Tensorflow https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077

Python Chatbot Project – Learn to build your first chatbot using NLTK & Keras https://data-flair.training/blogs/python-chatbot-project/

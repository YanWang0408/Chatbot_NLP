# Chatbot_NLP

Deep learning, NLTK

Data source: http://files.pushshift.io/reddit/comments/

Data format: jason file

Data preprocessing:

1. create a sqlite database to store interested data.
2. format data: normalize the comments and to convert the newline character to a word
3. data acceptable criteria:Len(data) between 1 and 50. because avg(len(data) is 50.
4. extract useful information from original json file: parent_id, body, cretaed_utc, score, id, subreddit..
5. 


Inspired by:
https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077

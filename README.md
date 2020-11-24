# Chatbot_NLP

Deep learning, NLTK

Data source: http://files.pushshift.io/reddit/comments/

Data format: jason file

Data preprocessing:
1. create a sqlite database to store data.
2. format data: new lines, utf-8 encoding
3. data acceptable criteria:Len(data) between 1 and 50. because avg(len(data) is 50.
4. extract useful information from original json file: parent_id, body, cretaed_utc, score, id, subreddit..
5. 

# save this as app.py
from flask import Flask, request, render_template, url_for
import re
import numpy as np
import pandas as pd
import community
import pandas as pd
import networkx as nx
import ast
import os
from collections import Counter
import csv
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
import ssl
import json
try:
     _create_unverified_https_context =     ssl._create_unverified_context
except AttributeError:
     pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from collections import Counter
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

app = Flask(__name__)
PATH = os.getcwd()

alay_dict = pd.read_csv(PATH+'/riset-digital/dataset/new_kamusalay.csv', encoding='latin-1', header=None)
alay_dict = alay_dict.rename(columns={0: 'original',
                                      1: 'replacement'})
stopwords_id = stopwords.words('indonesian')
stopwords_en = stopwords.words('english')

tt = TweetTokenizer()

model = 0

def tokenize_tweet(text):
    return " ".join(tt.tokenize(text))


def remove_unnecessary_char(text):
    text = re.sub("\[USERNAME\]", " ", text)
    text = re.sub("\[URL\]", " ", text)
    text = re.sub("\[SENSITIVE-NO\]", " ", text)
    text = re.sub('  +', ' ', text)
    return text


def preprocess_tweet(text):
    text = re.sub('\n', ' ', text)  # Remove every '\n'
    # text = re.sub('rt',' ',text) # Remove every retweet symbol
    text = re.sub('^(\@\w+ ?)+', ' ', text)
    text = re.sub(r'\@\w+', ' ', text)  # Remove every username
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', text)  # Remove every URL
    text = re.sub('/', ' ', text)
    # text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('  +', ' ', text)  # Remove extra spaces
    return text


alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
alay_dict_map.update({  # add some specific dictionary here
    # "pks"   : "pencegahan kekerasan seksual",
    # "p-ks"  : "pencegahan kekerasan seksual",
    # "pkl"    : "pedagang kaki lima"
})


def normalize_alay(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])


def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    return text


def remove_stopword(text):
    text = ' '.join(['' if word in stopwords_id else word for word in text.split(' ')])
    text = ' '.join(['' if word in stopwords_en else word for word in text.split(' ')])
    text = re.sub('  +', ' ', text)
    text = text.strip()
    return text


def preprocess(text, alay=False, tweet=False):
    if (tweet):
        text = preprocess_tweet(text)
    text = remove_unnecessary_char(text)
    text = text.lower()
    text = tokenize_tweet(text)
    if (alay):
        text = normalize_alay(text)
    return text


# Cleaning the tweets
def clean_tweet(text):
  df_clean = pd.DataFrame()
  df_clean['tweet'] = text
  # p.set_options(p.OPT.MENTION, p.OPT.EMOJI, p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.URL)
  # clean_text = [p.clean(a) for a in text.apply(str)]
  # # print(clean_text[:10])
  # df_clean['clean'] = clean_text
  # df_clean["sliced_text"] = [s[r[0]:r[1]] for s,r in zip(df_clean["full_text"], df_clean["display_text_range"])]
  clean_text = df_clean['tweet'].apply(str).apply(preprocess, args = (True,True,))
  df_clean['clean'] = clean_text.replace("[^a-zA-Z#]", " ")
  df_clean['no_stopword_text'] = clean_text.apply(remove_nonaplhanumeric).apply(remove_stopword)

  return(df_clean[['clean', 'no_stopword_text']])

def create_model():
    # Start Create Model
    train = pd.read_csv(PATH + '/indonlu/dataset/emot_emotion-twitter/train_preprocess.csv')
    test = pd.read_csv(PATH + '/indonlu/dataset/emot_emotion-twitter/valid_preprocess.csv')

    train[['clean', 'no_stopword_text']] = clean_tweet(train['tweet'])
    test[['clean', 'no_stopword_text']] = clean_tweet(test['tweet'])

    data = pd.concat([train, test])

    tfidf_vect = TfidfVectorizer(max_features=5000)
    tfidf_vect.fit(data['no_stopword_text'])
    train_X_tfidf = tfidf_vect.transform(train['tweet'])
    test_X_tfidf = tfidf_vect.transform(test['label'])

    model = SVC(kernel='linear')
    model.fit(train_X_tfidf, train['label'])
    predictions_SVM = model.predict(test_X_tfidf)
    test['Prediction'] = predictions_SVM

    pickle.dump(model, open(PATH + '/emot_tf-idf_model.sav', 'wb'))
    pickle.dump(tfidf_vect, open(PATH + "/emot_tfidf.pickle", "wb"))
    # End Create Model

def scrap(count, query, since, until):
    global model
    KEYWORD = query
    # Setting variables to be used in format string command below
    tweet_count = count
    text_query = KEYWORD + " lang:id include:nativeretweets"  # kata kunci yang digunakan untuk search di twitter
    PROJECT_NAME = "data"
    since_date = since
    until_date = until

    # Using OS library to call CLI commands in Python
    os.system(
        "snscrape --jsonl --max-results {} twitter-search '{} since:{} until:{}'> {}_scrap.json".format(tweet_count,
                                                                                                        text_query,
                                                                                                        since_date,
                                                                                                        until_date,
                                                                                                        PROJECT_NAME))
    data_df = pd.read_json(f'{PROJECT_NAME}_scrap.json', lines=True)

    # get username twitter from 'user'
    users = []
    # dataframe['column_name'] = dataframe['column_name'].fillna('').apply(str)
    user_prop = data_df['user'].fillna('').apply(str)
    i = 0
    for x in user_prop:
        i += 1
        if x != '' and re.search('({.+})', x) != None:
            dicti = ast.literal_eval(re.search('({.+})', x).group(0))
            users.append(dicti['username'])
        else:
            users.append('')

    data_df['username'] = users

    data_df[['clean', 'no_stopword_text']] = clean_tweet(data_df['content'])

    # Create Model
    if model == 0:
        create_model()
        model = 1

    loaded_tfidf = pickle.load(open(PATH + '/emot_tfidf.pickle', 'rb'))
    loaded_model = pickle.load(open(PATH + '/emot_tf-idf_model.sav', 'rb'))

    # load dataset
    test_data = data_df

    # some preprocessing and setup
    test_data['no_stopword_text'].fillna('0', inplace=True)
    X_tfidf = loaded_tfidf.transform(test_data['no_stopword_text'])  # TF-IDF

    # Proses Pengujian
    predictions_SVM = loaded_model.predict(X_tfidf)
    test_data['prediction'] = predictions_SVM

    most_tweet = test_data[['username', 'url','likeCount', 'retweetCount']].sort_values('retweetCount',ascending=False).head(10)

    word_count = Counter(" ".join(test_data['no_stopword_text']).split()).most_common(10)
    frequency = pd.DataFrame(word_count, columns=['Word', 'Frequency']).to_json(orient="columns")
    frequency_word = list(json.loads(frequency)['Word'].values())
    frequency_freq = list(json.loads(frequency)['Frequency'].values())

    hashtag = pd.notnull(test_data['hashtags'])
    ht_data = test_data[hashtag]
    ht_data['hashtags'] = ht_data['hashtags'].str.join(" ")
    word_text = " ".join(ht_data['hashtags'])
    emotions = data_df["prediction"].value_counts().to_json(orient="records")
    print(most_tweet.values.tolist())
    output_data = {}
    output_data['most_retweet'] = most_tweet.values.tolist()
    output_data['frequency_word'] = frequency_word
    output_data['frequency_freq'] = frequency_freq
    output_data['hashtag_wordcloud'] = word_text
    output_data['emotions'] = emotions

    return output_data



@app.route('/', methods=['GET', 'POST'])
def index():  # put application's code here
    status = 0
    output = ''
    # If request method is POST, here
    if request.method == 'POST':
        status = 1
        form_data = request.form
        query = form_data.get('keyword')
        count = form_data.get('tweet_count')
        since = form_data.get('since')
        until = form_data.get('until')
        output_data = scrap(count,query,since,until)
        return render_template('index.html', output=output, status=status, output_data = output_data)
    # If request method is GET, here
    else:
        return render_template('index.html', status=status)
    # return render_template('index.html', status=status)
if __name__ == '__main__':
    app.run(port=8000)
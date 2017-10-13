from __future__ import print_function

import json

import numpy as np
import pandas as pd
import string
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler

## For Topic Modeling
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *

class Extractor:

    def load_data(self, filename):
        posts_list = []
        posts_file = open(filename, "r")
        for line in posts_file:
            post = json.loads(line)
            posts_list.append(post)
        return posts_list


    def unpack_insights(self, post):
        insights = post['insights']
        new_insights = {}
        for insight in insights:
            value = insights[insight]['values'][0]['value']
            if isinstance(value, dict):
                for sub_ky in value:
                    new_ky = '{}_{}'.format(insight, sub_ky)
                    post[new_ky]=value[sub_ky]
            else:
                post[insight]=value
        del post['insights']
        return post

    def extract(self, posts_filename):
        posts = self.load_data(posts_filename)
        post = posts[0]
        featurenames_ex_insights = [ky for ky in post.keys() if ky!='insights' ]
        posts = [self.unpack_insights(post) for post in posts]
        insights_featurenames = [ky for ky in post.keys() if ky not in featurenames_ex_insights]
        label_names = ['post_impressions_organic_unique', 'name', 'message', 'description']
        posts_df = pd.DataFrame(posts)
        posts_df = posts_df[label_names]

        feature_names = ['name', 'message', 'description']
        posts_df[feature_names] = posts_df[feature_names].fillna('')
        posts_df['log_piou'] = np.log(posts_df['post_impressions_organic_unique'])
        posts_df['name_message_description'] =  posts_df['name'] + ' '+ posts_df['message'] + ' ' + posts_df['description']
        posts_df = posts_df[['log_piou', 'name_message_description']]

        return posts_df





class TopicModelTransformer:
    def __init__(self):

        self._stop = set(stopwords.words('english'))
        self._exclude = set(string.punctuation)
        self._lemma = WordNetLemmatizer()



    def clean(self, doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in self._stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in self._exclude)
        normalized = " ".join(self._lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    def display_topics(self, model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic {:d}:".format(topic_idx))
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))

    def transform(self, posts_df, num_topics, no_top_words):

        posts_df['name_message_description'] = posts_df['name_message_description'].apply(lambda x: self.clean(x))
        documents = posts_df['name_message_description']
        num_features = 10000

        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
        tf = tf_vectorizer.fit_transform(documents)
        tf_feature_names = tf_vectorizer.get_feature_names()
        lda = LatentDirichletAllocation(n_topics=num_topics,
                                              max_iter=5,
                                              learning_method='online',
                                              learning_offset=50.,
                                              random_state=0).fit(tf)

        self.display_topics(lda, tf_feature_names, no_top_words)

        topic_arrs = lda.transform(tf_vectorizer.transform(documents))
        topic_names = []
        for i in range(num_topics):
            topic_name = 'topic_{}'.format(str(i))
            topic_names.append(topic_name)

        X = pd.DataFrame(topic_arrs, columns=topic_names)
        X.fillna(X.mean())
        posts_df.fillna(posts_df.mean())
        Y = posts_df['log_piou']

        return X, Y

class Validator:
    def validate(self, X, Y):
        # Model Training & Evaluation
        M = len(Y)
        num_train = int(M*.8)
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train = X[-num_train:]
        X_train = x_scaler.fit_transform(X_train)

        X_test = X[:-num_train]
        X_test = x_scaler.transform(X_test)

        Y_train = Y[-num_train:]
        Y_train = y_scaler.fit_transform(Y_train.reshape(-1,1))

        Y_test = Y[:-num_train]
        Y_test = y_scaler.transform(Y_test.reshape(-1,1))

        model = RandomForestRegressor().fit(X_train, Y_train)

        y_pred = model.predict(X_test)

        print('explained_variance', explained_variance_score(Y_test, y_pred))
        print('neg_mean_absolute_error', mean_absolute_error(Y_test, y_pred) )
        print('neg_mean_squared_error',  mean_squared_error(Y_test, y_pred))
        print('neg_median_absolute_error', median_absolute_error(Y_test, y_pred))
        print('r2', r2_score(Y_test, y_pred))



posts_filename = 'posts.json'
posts_df = Extractor().extract(posts_filename)
X, Y = TopicModelTransformer().transform(posts_df, 16, 10)
Validator().validate(X, Y)


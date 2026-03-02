

import requests
import pandas as pd
from plotnine import *


##sentiment analysis of NotreDame subreddits vs national subreddits after big events
##or notre dame vs other team subbreddit
## Ohio state 2022, Marcus Freeman fight, Left out of playoffs 2025, 
##Super regional vs tennesee , Orange bowl vs penn state , 2016 S16vs wisconsin potential topics

def get_reddit_posts_comments(subreddit, search, total_posts=500, comments_per_post=50, start_date=None, end_date=None):
    post_url = "https://arctic-shift.photon-reddit.com/api/posts/search"
    comment_url = "https://arctic-shift.photon-reddit.com/api/comments/search"
    
    all_posts = []
    after = start_date
    while len(all_posts) < total_posts:
        params = {"subreddit": subreddit, "query": search, "sort": "asc",
                  "limit": min(100, total_posts - len(all_posts)), ## get around 100 post API limit
                  "before": end_date, "after": after}
        
        data = requests.get(post_url, params=params).json().get("data", [])
        if not data:
            break
        all_posts.extend(data)
        after = data[-1]["created_utc"]
        if len(data) < 100:
            break

    posts_df = pd.DataFrame(all_posts)[['id', 'title', 'author', 'score', 'num_comments', 'created_utc', 'selftext']]

    all_comments = []
    for post_id in posts_df['id']:
        data = requests.get(comment_url, params={"link_id": post_id, "limit": comments_per_post}).json().get("data", [])
        if data:
            df = pd.DataFrame(data)
            df['post_id'] = post_id
            all_comments.append(df)

    if not all_comments:
        return posts_df

    comments_df = (pd.concat(all_comments, ignore_index=True)
                   [['post_id', 'author', 'body', 'score', 'created_utc']]
                   .rename(columns={'author': 'comment_author', 'score': 'comment_score', 
                                    'created_utc': 'comment_created_utc'}))

    return posts_df.merge(comments_df, left_on='id', right_on='post_id', how='left')


## left out of playoff 2025
df_cfb = get_reddit_posts_comments(subreddit="CFB", search="Notre Dame", total_posts=100, 
start_date=1765065600,end_date=1765151999)
df_nd = get_reddit_posts_comments(subreddit="notredamefootball", search="Notre Dame", total_posts=400,
comments_per_post=100, start_date=1764997200,end_date=1765256399)

##2025 NIU loss september 7th-8th for CFB 7th-10th for NDFootball
'''
df_cfb = get_reddit_posts_comments(subreddit="CFB", search="Notre Dame", total_posts=200,comments_per_post=100, 
start_date= 1725667200 ,end_date=1725839999)
df_nd = get_reddit_posts_comments(subreddit="notredamefootball", search="Notre Dame", total_posts=600,
comments_per_post=100, start_date= 1725667200 ,end_date=1725983999)'''

df_cfb['source'] = 'CFB'
df_nd['source'] = 'ND_Football'
df_combined = pd.concat([df_cfb, df_nd], ignore_index=True)

##comment cleaning
import html
import re
import emoji
def clean_comment(comment):
    if comment is None or comment != comment or not str(comment).strip() or len(comment.strip())<15:
        return ''
    post_filter=['[deleted]', '[removed]',"This post has been removed", "I am a bot"]
    if [phrase for phrase in post_filter if phrase in comment]:
        return ''
    comment = html.unescape(comment)
    ## remove quoted parts of comment    
    comment=re.sub(r'^>.*$', '', comment, flags=re.MULTILINE)
    # remove URLs
    comment = re.sub(r'http\S+|www\.\S+', '', comment)
   ##remove \xa0 reddit character
    comment = comment.replace('\xa0', ' ')
    ##remove new line charatcer
    comment = comment.replace('\n', ' ')
    ##Remove extra whitespace
    comment = re.sub(r'\s+', ' ', comment)
    return comment.strip()


df_combined['cleaned_comment']=df_combined['body'].apply(clean_comment)
##filter out empty comments
df_combined= df_combined[df_combined['cleaned_comment'].str.strip() != '']
df_combined = df_combined.drop_duplicates(subset='cleaned_comment').reset_index(drop=True)



## sentiment analysis 
from textblob import TextBlob
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('spacytextblob')

def get_sentiment(comments):
    polarity     = []
    subjectivity = []
    
    for doc in nlp.pipe(comments, batch_size=64):
        polarity.append(doc._.blob.polarity)
        subjectivity.append(doc._.blob.subjectivity)
    
    return polarity, subjectivity

df_combined['polarity'], df_combined['subjectivity'] = get_sentiment(df_combined['cleaned_comment'])

polarity_dense=(
    ggplot(df_combined)
    +aes(x='polarity', fill='source')
    +geom_density(alpha=0.4)
      + theme(                                             
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_blank(),
        panel_background=element_blank()
    )
    + labs(
        title='Polarity Density Plot: CFB vs Notre Dame Subreddit',
        x='Polarity',
        y='Density',
        fill='Subreddit'
    )
)

polarity_dense
subjectivity_dense=(
    ggplot(df_combined)
    +aes(x='polarity', y='subjectivity', fill='source')
    +geom_point(alpha=0.4)
          + theme(                                             
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_blank(),
        panel_background=element_blank()
    )
    + labs(
        title='Polarity vs Subjectivity Scatterplot: CFB vs Notre Dame Subreddit',
        x='Polarity',
        y='Density',
        fill='Subreddit'
    )
)
subjectivity_dense
polarity_dense
from transformers import pipeline
import torch

def emotion_sentiment(comments):
    emotion=pipeline('sentiment-analysis', model="j-hartmann/emotion-english-distilroberta-base",
    top_k=2)
    emotion_results= emotion(comments.to_list(),truncation=True, max_length=512)
    emotion_1, emotion_1_score = [], []
    emotion_2, emotion_2_score = [], []
    for result in emotion_results:
        emotion_1.append(result[0]['label']);  emotion_1_score.append(round(result[0]['score'], 4))
        emotion_2.append(result[1]['label']);  emotion_2_score.append(round(result[1]['score'], 4))
    return   emotion_1, emotion_1_score, emotion_2, emotion_2_score

(df_combined['emotion_1'], df_combined['emotion_1_score'], df_combined['emotion_2'], df_combined['emotion_2_score'],
 ) = emotion_sentiment(df_combined['cleaned_comment'])

boxplot = (
    ggplot(df_combined)
    + aes(x='emotion_1', y='comment_score', fill='source')
    + geom_boxplot(outlier_shape='') ## hide outliers
    + geom_hline(yintercept=0, linetype='dashed', color='gray', size=0.5)
    + theme(                                             
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_blank(),
        panel_background=element_blank()
    )
    + labs(
        title='Comment Score by Emotion: CFB vs Notre Dame Subreddit',
        x='Emotion',
        y='Comment Score',
        fill='Subreddit'
    ))
boxplot
## topic modeling 
import numpy as np
import lda
import nltk
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pprint as pprint
import requests
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from gensim.models.coherencemodel import CoherenceModel


stop_words = nltk.corpus.stopwords.words('english')


def sent_to_words(comments):
    for comment in comments:
        yield(gensim.utils.simple_preprocess(str(comment), deacc = True))  

comment_words=list(sent_to_words(df_combined['cleaned_comment']))

                
custom_stopwords = ['game','team','play','played','nd', 'notre', 'dame', 'irish', 
    'football', 'guy', 'year', 'week', 'saturday', 'win', 'loss', 
    'think', 'see', 'watch', 'going', 'got', 'know', 'really','quarter',
    'south', 'bend','college','coach','ball', 'season', 'year', 'commerical',
    'us','people','someone', 'made',"like", "teams","fan","get","also","games",
    ]
for x in custom_stopwords:
    stop_words.append(x)


def lemmatization(comments, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in comments:
        doc = nlp(" ".join(sent)) 
        current_sentence_lemmas = []
        for token in doc:
            if token.pos_ in allowed_postags:
                current_sentence_lemmas.append(token.lemma_)
        texts_out.append(current_sentence_lemmas)
    return texts_out


def remove_stopwords(comments):
    final_output = []
    for doc in comments:
        doc_words = []
        for word in doc:
            cleaned_word = gensim.utils.simple_preprocess(str(word))
            if cleaned_word:  
                word_str = cleaned_word[0]
                if word_str not in stop_words:
                    doc_words.append(word_str)
        final_output.append(doc_words)
    return final_output

bigram = gensim.models.Phrases(
comment_words, min_count=5, threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)

def make_bigrams(comments):
    return [bigram_mod[comment] for comment in comments]


biagrams= make_bigrams(comment_words)

data_lemmatized = lemmatization(biagrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
data_lemmatized=remove_stopwords(data_lemmatized)
id2word = gensim.corpora.Dictionary(data_lemmatized)
texts = data_lemmatized
corpus = []
for text in texts:
    corpus.append(id2word.doc2bow(text))

lda_model = gensim.models.ldamodel.LdaModel(
  corpus=corpus,
  id2word=id2word,
  num_topics=5, 
  random_state=100, 
  update_every=1, 
  chunksize=100, 
  passes=10, 
  alpha='auto',
  per_word_topics=True 
)

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

pyLDAvis.enable_notebook()
vis = gensimvis.prepare(lda_model, corpus, id2word, mds='mmds')
pyLDAvis.display(vis)



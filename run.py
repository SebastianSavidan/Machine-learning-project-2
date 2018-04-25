# run.py
#''' Final Code for Text Sentiment Classification'''
# Kaggle Group : [Sentiment Analysis Joke]
# Sebastian Savidan, Jean Gschwind & Tristan Besson


# All imports
import pandas as pd
import numpy as np
import re

from helpers import *

from nltk.tokenize import TweetTokenizer

from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models.word2vec import Word2Vec
# If error -> pip install --upgrade gensim

from tqdm import tqdm, tnrange
# If error -> pip install tqdm

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
# If error -> pip install -U nltk
nltk.download('wordnet') #Uncomment this line if first time using 'wordnet' corpus
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import SnowballStemmer
from nltk.corpus import stopwords
lmtzr = WordNetLemmatizer()
ps = PorterStemmer()
sb = SnowballStemmer("english")

# __________________________Helper functions__________________________

def convert_tag_to_pos(tag): #(for lemmatizer)
    """ Convert the tag given by pos_tag() into something lemmatize can understand """
    if tag in pos_dict.keys():
        return pos_dict[tag]
    else:
        return 'n' #Default value

def get_tokenized_tweet(tweet): 
    """ Clean tweet (remove punctations and numbers and tokenize it)"""
    # Remove all punctation & all numbers
    tweet = re.sub('[^A-Za-z ]+','', tweet)
    
    # Remove user & url
    #tweet = re.sub('user', '', tweet)                            
    #tweet = re.sub('url', '', tweet)
    
    tokens = TweetTokenizer().tokenize(tweet)
    filtered_sentence = []
 
    # Stop words filtering
    #for w in tokens:
    #    if w not in stop_words:
    #        filtered_sentence.append(w)
    #tokens = filtered_sentence
    #del filtered_sentence
   
    # Lemmatization 
    #tokens = [lmtzr.lemmatize(word,convert_tag_to_pos(tag)) for word,tag in tagged]
    
    # Stemming
    tokens = [sb.stem(word) for word in tokens]

    return tokens

def get_taggedDocument(X): 
    """ Prepare taggedDocument class from tweets for model"""
    taggedDocument = []
    
    for index in tqdm(range(len(X))):
        words = X[index]
        tags = "TRAIN_" + str(index)
        
        taggedDocument.append(TaggedDocument(words, tags))
        
    return taggedDocument

def convert_tweet_to_vector(tweet, size, w2v_model, tf_idf):
    """ Convert a tweet in a vector based on the w2v model"""
    vector = np.zeros(size).reshape((1, size))
    
    for word in tweet:
        try:
            vector += w2v_model[word].reshape((1, size)) * tf_idf[word]
        except KeyError:
            continue
        
    return vector

def get_vectors(tweets, w2v_model, tf_idf):
    """ Create the matrix associated to a set of tweets"""
    n_dim=250
    all_vectors = [convert_tweet_to_vector(tweet, n_dim, w2v_model, tf_idf) for tweet in tqdm(tweets)]
    
    return np.concatenate(all_vectors)


# __________________________Main function__________________________ 

def main():
    
    ## _____Dataset Handling____

    print("Loading Datasets")
    # Import positive and negative tweets 
    tweet_pos_df = pd.read_csv('twitter-datasets/train_pos_full.txt', 
                               names=['text'], delimiter="\t", header=None)
    tweet_pos_df['sentiment'] = 1

    tweet_neg_df = pd.read_csv('twitter-datasets/train_neg_full.txt', 
                               names=['text'], delimiter="\t", header=None)
    tweet_neg_df['sentiment'] = -1

    # Create the general dataframe
    tweets_df = tweet_pos_df.append(tweet_neg_df)
    tweets_df = tweets_df.reset_index(drop=True)
    del tweet_pos_df, tweet_neg_df

    
    ## _____Dataset Processing____
    print("Processing Datasets")
    tweets_df = tweets_df.sample(frac=1).reset_index(drop=True)
    
    pos_dict = {'N' : 'n', 'V' : 'v', 'J' : 'a', 'S' : 's', 'R' : 'r'}

    stop_words = set(stopwords.words('english'))
    
    tweets_df['tokenized text'] = tweets_df['text'].map(get_tokenized_tweet)
    
    
    
    # Split data in test and train set
    X_train, X_test, y_train, y_test = train_test_split(tweets_df['tokenized text'], 
                                                        tweets_df['sentiment'],
                                                        test_size=0.25,
                                                        random_state=42)

    del tweets_df
    
    
    ## _____Word2Vec Model Creation____
    print("Word2vec model creation")
    taggedDocument = get_taggedDocument(list(X_train))
    all_words = [x.words for x in taggedDocument]
    n_dim = 250

    # Creation of the model
    w2v_model = Word2Vec(size=n_dim, min_count=2,workers = 4) # Test with different min_count !

    # Build vocab
    w2v_model.build_vocab(all_words)

    # Train the model
    print("Train w2v model...")
    w2v_model.train(all_words,
                    total_examples=w2v_model.corpus_count,
                    epochs=w2v_model.iter)

    
    ## _____TF-IDF Matrix____

    print("TF-IDF Matrix Creation")
    # Create tf-idf matrix
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    matrix = vectorizer.fit_transform(all_words)
    tf_idf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    
    # Compute the vectors matrix from tweets
    X_train_w2v = get_vectors(X_train, w2v_model, tf_idf)
    del X_train
    X_test_w2v  = get_vectors(X_test, w2v_model, tf_idf)
    del X_test

    
    ## _____Classifier____
    print("Classification")
    size = (50, 50, 50)
    classifierPercZ = MLPClassifier(solver='adam', alpha=1e-7, hidden_layer_sizes = size, random_state=1)
    classifierPercZ.fit(X_train_w2v, y_train)

    print("The score on the test set is:", classifierPercZ.score(X_test_w2v, y_test))

    
    ## _____Prediction File Creation____
    # Import data
    tweet_unlabelized_df = pd.read_csv('twitter-datasets/test_data.txt', 
                                   names=['text'], delimiter="\t", header=None)
    tweet_unlabelized_index = tweet_unlabelized_df.index.values + 1

    # Process data
    X_unlabelized = tweet_unlabelized_df['text'].map(get_tokenized_tweet)
    X_unlabelized_w2v = get_vectors(X_unlabelized, w2v_model, tf_idf)

    # Make predictions
    y_unlabelized = classifierPercZ.predict(X_unlabelized_w2v)

    # Create submission file
    create_csv_submission(tweet_unlabelized_index, y_unlabelized, "text_class_submission_top.csv")
    print("Submission file created !")


if __name__ == '__main__':
    main() 
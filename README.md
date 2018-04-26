# Project Text Sentiment Classification

Desclaimer:
This is the second project of the Machine Learning course from EPFL.
You can clone and use this work.
This project was a short one so it might have some mistakes or can be incomplete.
Authors: Jean Gschwind, Tristan Besson, Sebastian Savidan

## Files Description
_run.py_

$ python run.py

Our script for sentence classification. It uses the full twitter dataset to train, and will output a submission file with the predicitons of the testing data. 

The script first removes special characters from the tweets, tokenise the sentences and then apply a stemmer to them. We then train a word2vec model with all the words of our dataset (with a min occurence of 1000 as we use the full dataset). We then transform each word to a vector,add the TF-IDF weights and contruct the sentences vectors. Then a perceptron is then used to separate the tweets into two classes (positive tweets and negative tweets).

_run.ipynb_

Same as run.py, but presented in a jupyter notebook with more comments. 

_helpers.py_

Contains the function for the submission file creation.

## How to run the classifier

- Put the training and testing dataset in a folder called data next to the scripts folder twitter-datasets

- Open a terminal, navigate to the scripts directory, and type: jupyter notebook (you need to have anaconda installed on your computer)

If not functionning, type the following commands in a terminal:

python3 -m pip install --upgrade pip
python3 -m pip install jupyter

## Needed Libraries

- gensim
pip install --upgrade gensim

- tqdm
pip install tqdm

- nltk
pip install -U nltk

## Saved Results Location
The corresponding prediction csv file will be saved in the same folder as the script and will be called "text_class_submission_top.csv"

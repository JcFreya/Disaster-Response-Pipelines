import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle

import nltk
nltk.download(['punkt','wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin


def load_data(database_filepath):
    """
    Load data from database
    Args:
        database_filepath: String location of database file
    Returns:
        X: numpy.ndarray features, disaster message
        y: numpy.ndarray target variable, disaster message's categories
        categories: list, disaster category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    categories = df.columns[4:]
    X = df.message.values
    y = df[categories].values
    
    return X, y, categories
    

def tokenize(text):
    """
    Tokenization function to process text data
    Args:
        text: String. disaster message.
    Returns:
        clean_tokens: list. token list from text message.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if (first_tag in ['VB', 'VBP'] or first_word == 'RT'):
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
    
def build_model():
    """
    Build the pipeline and gridsearch parameter, and return the model
    Retunrs:
        cv: pipeline with gridsearch parameters
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1))))    
    ])

    parameters = {
        'features__text_pipeline__vect__max_df': (0.75, 1.0),
        'clf__estimator__learning_rate': [0.1, 0.3]
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose=2)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predict the test data using trained model
    Args:
        model: sklearn.model_selection.GridSearchCV.  It contains a sklearn estimator.
        X_test: numpy.ndarray. Disaster messages.
        Y_test: numpy.ndarray. Disaster categories for each messages
        category_names: Disaster category names.
    """
    # predict on test data
    y_pred = model.predict(X_test)
    
    for i in range(0, len(category_names)):
        print('--------------------\n')
        print('Category: {}\n'.format(category_names.values[i]))
        print(classification_report(Y_test[:,i],y_pred[:,i]))
    

def save_model(model, model_filepath):
    """
    Export the model as a pickle file
    Args:
        model: sklearn.model_selection.GridSearchCV.
        model_filepath: String. location to save the trained model
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        print(model)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
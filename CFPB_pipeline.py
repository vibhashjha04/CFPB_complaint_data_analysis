import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from pprint import pprint
from time import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Fetch CFPB Data , Filter with small samples
df_data = pd.read_csv('complaints.csv',usecols=["Product","Consumer complaint narrative","Company"],header=0,low_memory=False).dropna()

df_data = df_data.loc[df_data['Company'] == 'BANK OF AMERICA, NATIONAL ASSOCIATION']
df = df_data.head(100)
#print(len(df),df.size)

#Prepare training matrix for further steps.
df_attr = df['Consumer complaint narrative']
df_target = df['Product']

# Prepare test matrix data set for later prediction
df_t = df_data.tail(100)
df_t_attr = df_t['Consumer complaint narrative']
df_t_target = df_t['Product']

# Tune hyperparameters through gridsearch methods
# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__max_iter': (20,),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    # 'clf__max_iter': (10, 50, 80),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1,cv=2)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    #grid_search.fit(data.data, data.target)
    grid_search.fit(df_attr,df_target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()

    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # Build classifier model using best parameters set. Remove english stop words & only select words token
    # This method uses pipeline..

    text_clf = Pipeline([
        ('vect',
         CountVectorizer(stop_words='english', max_df=0.5, ngram_range=(1, 2), token_pattern=r'(?u)\b[A-Za-z]+\b')),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(alpha=1e-5, max_iter=20, penalty='elasticnet')),
    ])

    text_clf.fit(df_attr, df_target)
    predictions = text_clf.predict(df_t_attr)

    print('Test accuracy is {}'.format(accuracy_score(df_t_target, predictions)))

    # Building same model without pipeline method to know intermediate steps..

    vectorizer = CountVectorizer(stop_words='english', max_df=0.5, ngram_range=(1, 2),
                                 token_pattern=r'(?u)\b[A-Za-z]+\b')
    X_v = vectorizer.fit_transform(df_attr).toarray()
    # print(vectorizer.get_feature_names())
    # print(X_v)

    transformer = TfidfTransformer()
    X_t = transformer.fit_transform(X_v).toarray()
    # print(X_t)

    le = LabelEncoder()
    y_t = le.fit_transform(df_target)
    # print(y_t)

    classifier = SGDClassifier(penalty='elasticnet', alpha=1e-6, max_iter=20, random_state=3000)
    classifier.fit(X_t, y_t)

    predictions = classifier.predict(X_t)

    print('Test accuracy is {}'.format(accuracy_score(y_t, predictions)))

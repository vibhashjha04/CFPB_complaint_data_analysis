import pandas as pd
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder

# Fetch CFPB Data , Filter with small samples
df_data = pd.read_csv('complaints.csv',usecols=["Product","Consumer complaint narrative","Company"],header=0,low_memory=False).dropna()

#df_data = df_data.loc[df_data['Company'] == 'BANK OF AMERICA, NATIONAL ASSOCIATION']
df = df_data.head(50)
#print(len(df),df.size)
#print(df)

#Prepare training matrix for further steps.
df_attr = df['Consumer complaint narrative']
df_target = df['Product']

# Prepare test matrix data set for later prediction
df_t = df_data.tail(10)
df_t_attr = df_t['Consumer complaint narrative']
df_t_target = df_t['Product']

vectorizer = CountVectorizer(stop_words='english', max_df=0.5, ngram_range=(1, 2),
							 token_pattern=r'(?u)\b[A-Za-z]+\b')
X_v = vectorizer.fit_transform(df_attr).toarray()
# print(vectorizer.get_feature_names())
#print(X_v)

transformer = TfidfTransformer()
X_t = transformer.fit_transform(X_v).toarray()
#print(X_t)

le = LabelEncoder()
y_t = le.fit_transform(df_target)
#print(y_t)

X = X_t
y = y_t

'''
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
'''

#print (X)
#print (y)

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10, verbose=1)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
# make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(X.shape[0]):
	#print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
	print(predictions[i], y[i])
from sklearn.metrics import confusion_matrix
print(confusion_matrix(predictions, y))
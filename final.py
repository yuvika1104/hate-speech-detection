#importing libraries
import pandas as pd
import numpy as np
#reading the dataset
dataset=pd.read_csv('labeled_data.csv')
#assigning labels
dataset["labels"]= dataset["class"].map({0: "Hate Speech", 1: "Hate Speech", 2:"Not Hate Speech"})
#reducing the dataset to tweets and labels
dataset=dataset[["tweet","labels"]]
#print(dataset.head())
#libraries  for NLP
import re
import nltk
import string
#importing stopwords
#nltk.download('stopwords')
from nltk.corpus import stopwords
stopword=set(stopwords.words('english'))
#Stemmer
stemmer=nltk.SnowballStemmer('english',ignore_stopwords=True)
#function to clean the text
def clean(tweet):
  tweet= tweet.lower()
  tweet= re .sub('[.?]','', tweet)
  tweet= re.sub('<.?>+', '', tweet)
  tweet= re.sub('[%s]' % re.escape(string.punctuation), '', tweet)
  tweet= re.sub('\n', '', tweet)
  tweet= [word for word in tweet.split(' ')if word not in stopword]
  tweet= " ".join(tweet)
  tweet= [stemmer.stem(word) for word in tweet.split(' ')]
  tweet= " ".join(tweet)
  return tweet
#cleaning the dataset
dataset["tweet"] = dataset["tweet"].apply(clean)
#seperating the dataset
x= np.array(dataset["tweet"])
y= np.array(dataset["labels"])

from sklearn.feature_extraction.text import TfidfVectorizer 
# Forming the TF-IDF model
v = TfidfVectorizer()
v.fit(x)

def vectorize(train_data):
    X = v.transform(train_data)
    return X
X = vectorize(x)
#dividing the train and test set
from sklearn. model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X ,y, test_size=0.33, random_state= 42)

from sklearn. tree import DecisionTreeClassifier
model= DecisionTreeClassifier(random_state=100)
# Training the model
model.fit(X_train, y_train)
#making the pickle file
import pickle
pickle.dump(model,open('modelfile.pkl','wb'))
#testing the model
y_pred= model.predict (X_test)

#accuracy of model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

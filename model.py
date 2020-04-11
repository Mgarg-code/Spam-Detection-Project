# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

df= pd.read_csv('SMSSpamCollection', sep='\t', header=None)
df.columns = [ 'class', 'message']


# prepare the data for training by converting eht text data into vector form
cv=CountVectorizer()
data=cv.fit_transform(df['message'])
data=data.toarray()
data=pd.DataFrame(data)

#Splitting Training and Test Set
target=df['class']
x_train,x_test,y_train,y_test=train_test_split(data,target)
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.naive_bayes import MultinomialNB
nv=MultinomialNB()

#Fitting model with trainig data
nv.fit(x_train, y_train)

# Saving model to disk
pickle.dump(nv, open('model.pkl','wb'))
new_messeges = [
 "free free absolute free register today demo",
]
test=cv.transform(new_messeges).toarray()
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict(test))
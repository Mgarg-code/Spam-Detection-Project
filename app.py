import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI

    '''
    df= pd.read_csv('SMSSpamCollection', sep='\t', header=None)
    df.columns = [ 'class', 'message']
    cv=CountVectorizer()
    cv.fit(df['message'])
    new_messeges=request.form.get("text") 
    new=[]
    new.append(new_messeges)
    test=cv.transform(new).toarray()  

    output = model.predict(test)
    return render_template('index.html', prediction_text='your message should be a $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
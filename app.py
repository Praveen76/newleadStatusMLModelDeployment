import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template, url_for
#from model import encode
import pickle
import operator

app = Flask(__name__)
model = pickle.load(open('./RF_Model.sav','rb'))

enc = pickle.load(open('./encoder.pkl','rb'))
 

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    categorical_cols = ['LeadSourceGroup', 'LoanPurpose', 'ZipCode']
    
    int_features = [x for x in request.form.values()]
    print('int_features',int_features)
    Catindices = [0,1,2]
    Numindices=[3,4,5,6]
    catVar = [element for i, element in enumerate(int_features) if i in Catindices]
    catVar
    numVar = [element for i, element in enumerate(int_features) if i in Numindices]
    catVar=[catVar]
    print('catVar',catVar)
    print('numVar',numVar)
    
    df=pd.DataFrame(catVar,columns=categorical_cols)
    Encoded = enc.transform(df)
    Encoded=Encoded.tolist()[0] 
    print('Encoded',Encoded)
    print('Encoded',len(Encoded))
    AllFeatures=Encoded+numVar
     
    int_features=[int_features]
    print('int_features----->',int_features)
    
    final_features = [np.array(AllFeatures)]
    prediction = model.predict_proba(final_features)
    lst=list(prediction[:,1])
    predPrctg=[x*100 for x in lst]
    print(prediction[0])

    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Predicted leads percentage for input datapoints : {} %".format(predPrctg[0]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict_proba([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)
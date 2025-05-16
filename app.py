import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
filename = 'prediction-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':

        oc = request.form['oc']
        de = request.form['de']
        st = request.form['st']
        hu = request.form['hu']
        si = request.form['si']
        dy = request.form['dy']
        po = request.form['po']
        se = request.form['se']
        inc = request.form['in']
        co = request.form['co']
        #oc = int(oc)
        de =  float(de)
        st = int(st)
        hu = int(hu)
        si = int(si)
        dy = int(dy)
        po = int(po)
        se = int(se)
        inc = int(inc)
        co = int(co)
        data = np.array([[de, st, hu, si, dy, po, se, inc,co]])
        print(data)
        my_prediction = classifier.predict(data)
        print(my_prediction[0])
        if my_prediction == 1:
            print('Bleaching Disease')
            result="Bleaching Disease"

        else:
            print('Non-Bleaching Disease')
            result="Non-Bleaching Disease"

    #output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Coral_Bleaching Result : {}'.format(result))



if __name__ == "__main__":
    app.run(debug=True)
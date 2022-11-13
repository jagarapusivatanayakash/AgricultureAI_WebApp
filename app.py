from flask import Flask, render_template, request, flash, redirect, Markup
import sys
import os
import glob
import re
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

app = Flask(__name__)

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from werkzeug.utils import secure_filename

# Model saved with Keras model.save()
MODEL_PATH ='models\disease.h5'
# Load your trained model
disease_model = load_model(MODEL_PATH)

fertilizer_model=pickle.load(open("models/fertilizer.pkl","rb"))
le_crop=pickle.load(open("models/le_crop.pkl","rb"))
le_soil=pickle.load(open("models/le_soil.pkl","rb"))
crop_model=pickle.load(open("models/crop.pkl","rb"))

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

def model_predict(img_path, disease_model):
    print(img_path)
    img = image.load_img(img_path)
    img = img.resize((224,224))
    #img = np.asarray(img)
    #img = img.reshape((1,36,36,1))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
    preds = disease_model.predict(x)
    preds=np.argmax(preds, axis=1)
    preds=disease_classes[preds[0].item()]
    return preds

@ app.route('/')
def home():
    title = 'Home'
    return render_template('home.html', title=title)

# render crop recommendation form page


@ app.route('/crop_recommend')
def crop_recommend():
    title = 'Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page``

@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)

@app.route('/disease', methods=['GET'])
def disease():
    title = 'disease'
    return render_template('disease.html',title=title)

@app.route("/crop_predict", methods = ['POST'])
def crop_predict():
    title = 'Crop Recommendation'
    try:
        if request.method == 'POST':
            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorous'])
            K = int(request.form['pottasium'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            temperature= float(request.form['temperature'])
            humidity= float(request.form['humidity'])
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_model.predict(data)
            final_prediction = my_prediction[0]
            return render_template('crop_result.html', prediction=final_prediction,title=title)
    except:
        return render_template("fertilizer_result.html",prediction ='Please enter valid Data',title=title)

@app.route("/fertilizer_predict", methods = ['POST'])
def fertilizer_predict():
    title = 'Fertilizer Suggestion'
    try:
        if request.method == 'POST':
            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorous'])
            K = int(request.form['pottasium'])
            temperature= float(request.form['temperature'])
            humidity= float(request.form['humidity'])
            moisture= float(request.form['moisture'])
            soiltype= str(request.form['soiltype'])
            croptype= str(request.form['croptype'])
            soiltype= le_soil.transform([soiltype])
            croptype= le_crop.transform([croptype])
            data = np.array([[temperature, humidity,moisture,soiltype[0],croptype[0],N,K,P]])
            my_prediction = fertilizer_model.predict(data)
            final_prediction = my_prediction[0]
            final_prediction = Markup(str(fertilizer_dic[final_prediction]))
            return render_template("fertilizer_result.html", prediction=final_prediction,title=title)
    except:
        return render_template("fertilizer_result.html", prediction= 'Please enter valid Data',title=title)

@app.route('/disease_predict', methods=['GET', 'POST'])
def disease_upload():
    title="disease"
    try:
        if request.method == 'POST':
            # Get the file from post request
            f = request.files['file']
            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            # Make prediction
            preds = model_predict(file_path, disease_model)
            preds = Markup(str(disease_dic[preds]))
            return render_template('disease_result.html', prediction=preds, title=title)
    except:
        return render_template('disease_result.html', prediction= 'Please enter valid Data', title=title)

if __name__ == '__main__':
    app.run(debug=False)
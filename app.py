from PIL import Image
import face_recognition
import os
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from werkzeug.utils import secure_filename
from phone_detection import *
from background_check import *

app = Flask(__name__)
model1= keras.models.load_model("Eye-Detection/eye_detection.h5")
model2= keras.models.load_model("Yawn-Detection/yawn_detection.h5")
model3= keras.models.load_model("Eye-Direction/eye_direction.h5")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods = ["POST", "GET"])
def predict():
    if request.method == "POST":
        f = request.files['img']
        filename = secure_filename(f.filename)
        f.save(os.path.join('/', filename))
        print(os.path.join('/', filename))
        img=image.load_img(os.path.join('/', filename),target_size=(64,64))

        #function to detect cell phone  
        phone = phone_detection(os.path.join('/', filename))

        #function to check surroundings
        env = background_check(os.path.join('/', filename))

        #function to detect yawn
        def yawn_detection():
            y=image.img_to_array(img)
            y=np.expand_dims(y,axis=0)
            pred1=np.argmax(model2.predict(y))
            if pred1==0:
                return True
            else:
                return False

        #function to detect eye gaze (open/closed)
        def eyegaze_detection():
            pic = face_recognition.load_image_file(os.path.join('/', filename))
            face_landmarks_list = face_recognition.face_landmarks(pic)
            
            eyes = []
            eyes.append(face_landmarks_list[0]['left_eye'])
            eyes.append(face_landmarks_list[0]['right_eye'])

            for eye in eyes:
                x_max = max([coordinate[0] for coordinate in eye])
                x_min = min([coordinate[0] for coordinate in eye])
                y_max = max([coordinate[1] for coordinate in eye])
                y_min = min([coordinate[1] for coordinate in eye])
                 
            x_range = x_max - x_min
            y_range = y_max - y_min

            if x_range > y_range:
                right = round(.5*x_range) + x_max
                left = x_min - round(.5*x_range)
                bottom = round(((right-left) - y_range))/2 + y_max
                top = y_min - round(((right-left) - y_range))/2
            else:
                bottom = round(.5*y_range) + y_max
                top = y_min - round(.5*y_range)
                right = round(((bottom-top) - x_range))/2 + x_max
                left = x_min - round(((bottom-top) - x_range))/2

            im = Image.open(os.path.join('/', filename))
            im = im.crop((left, top, right, bottom))

            im = im.resize((64,64))
            im.save(os.path.join('/', filename))

            x=image.img_to_array(img)
            x=np.expand_dims(x,axis=0)
            pred=np.argmax(model1.predict(x))

            if pred==0:
                return "closed"
            else:
                return "open"

        def eye_direction():
            x=image.img_to_array(img)
            x=np.expand_dims(x,axis=0)
            pred=np.argmax(model3.predict(x))

            if pred==1:
                return "right"
            else:
                return "left"

        return{
            "surroundings": env,
            "eyes": eyegaze_detection(),
            "gaze":eye_direction(),
            "yawn": yawn_detection(),
            "cellphone":phone
        }
       
        
if __name__ == "__main__":
    app.run(debug = True)
from flask import Flask, render_template, request
import cv2
import numpy as np
import pandas as pd
import os
from tensorflow.keras import backend as K
from keras.models import load_model
import h5py
from matplotlib import pyplot as plt
from scipy import misc
import PIL
from PIL import Image as im
import numpy
import copy
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


smooth=100

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)

model = load_model('custom2_brain_mri_seg.hdf5', custom_objects={'dice_coef': dice_coef , 'dice_coef_loss':                   
dice_coef_loss, 'iou' : iou, 'jac_distance' : jac_distance})

print(model.summary())



im_width = 256
im_height = 256


@app.route('/')
def home():
	return render_template('login.html')


@app.route('/submit_template'  , methods=['GET', 'POST'])
def submit_template():
	return render_template("index.html", data="hey")


@app.route("/prediction", methods=["POST"])
def prediction():
	for filename in os.listdir('static/'):
		if filename.startswith('bgz_'):  # not to remove other images
			os.remove('static/' + filename)


	img_pre = request.files['img']; 
	print(img_pre)
	img_pre.save("img_pre.jpg")
	img_pre = im.open("img_pre.jpg")
	img_pre.save('static/bga.png');


	del(img_pre)

	img = copy.copy('img_pre.jpg')

	img = (PIL.Image.open('static/bga.png'))
	
	img.save("img.jpg")
	img = cv2.imread("img.jpg")
	img = cv2.resize(img ,(im_height, im_width))
	img = img / 255
	img = img[np.newaxis, :, :, :]
	pred=model.predict(img)
	pred = np.squeeze(pred) > 0.5


	dataImg = im.fromarray((pred))
	dataImg.show();
	dataImg.save('static/bgz.png');
	

	



	return render_template("prediction.html", data='static/bgz.png')





if __name__ == "__main__":
	app.run(debug=True)

# Faizan Ahmed
# 2/11/2021


import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import keras
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import os,sys
import lime
from lime import lime_image

#model = tf.keras.models.load_model('COVID-TransferLearningModel-v1-200-3-5-10.h5')

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('COVID-TransferLearningModel-v2.h5')


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


st.write("""
         # COVID-19 Detection via Deep Learning
         """
         )

st.write("Automated detection of COVID-19 & Viral Pneumonia in Chest X-rays with Deep Learning")


file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


def import_and_predict(image_data, model):
    
    size = (200,200)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(200, 200), interpolation=cv2.INTER_CUBIC))/255.

    img_reshape = img_resize[np.newaxis,...]

    prediction = model.predict(img_reshape)
    y_classes = prediction.argmax(axis=-1)

    return prediction, y_classes
    

def transform_img_fn(img, path_list):
    
    out = []
    
    for img_path in path_list:
        img_ = image.load_img(img_path, target_size=(200, 200))
        x = image.img_to_array(img_)
        x = np.expand_dims(x, axis=0)
        x = x / 255
        x = x.reshape(1,200, 200,3)
    
    return x


def displayExplainations(model, img):
    
    explainer = lime_image.LimeImageExplainer()
    
    exp = explainer.explain_instance(img[0].astype('double'), model.predict, hide_color=0, num_samples=100)
    
    temp, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=False, hide_rest=False)
    
    fig = mark_boundaries(temp / 2 + 0.5, mask)
    plt.imsave('img\\test.png', fig,  dpi=2000)
    
    load_img = cv2.imread('img\\test.png')
    load_img = cv2.cvtColor(load_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Explaination', load_img)
    st.image(load_img, caption='Model Explanation', use_column_width=True)
    
    temp, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    
    fig = mark_boundaries(temp / 2 + 0.5, mask)
    plt.imsave('img\\test2.png', fig,  dpi=2000)
    
    load_img = cv2.imread('img\\test2.png')
    load_img = cv2.cvtColor(load_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Explaination', load_img)
    st.image(load_img, caption='Model Explanation 2', use_column_width=True)
    
    
if file is None:
    
    st.text("Please upload an image file")
    
else:
    
    uploadedImage = Image.open(file)
    filePath = 'C:\\Users\\faiza\\COVID-19 Deep Learning\\train\\' + file.name
    
    st.image(uploadedImage, use_column_width=True)
    prediction = import_and_predict(uploadedImage, model)[0]
    category = import_and_predict(uploadedImage, model)[1]
    
    if np.argmax(prediction) == 0:
        st.write("Prediction: COVID-19")
    elif np.argmax(prediction) == 1:
        st.write("Prediction: Normal")
    else:
        st.write("Prediction: Viral Pneumonia")
    
    st.text("Probability (0: COVID-19, 1: Normal, 2: Viral Pneumonia")
    st.write(prediction)
    
    img = transform_img_fn(uploadedImage, [filePath])
    displayExplainations(model, img)
    
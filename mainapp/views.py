from __future__ import division, print_function
from django.shortcuts import render
from rest_framework import generics
from rest_framework.response import Response
from .serializers import ImageSerializer

# coding=utf-8
import sys
import os
from io import BytesIO
import glob
import re
import numpy as np
import tensorflow as tf

# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Create your views here.
def model_predict_acne_face(img_path, model):
    img_bytes = BytesIO(img_path)
    img = tf.keras.preprocessing.image.load_img(img_bytes, target_size=(224, 224))

    # Preprocessing the image
    x = tf.keras.preprocessing.image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling      1111111bbb
    x = x / 225
    x = np.expand_dims(x, axis=0)

    # x = preprocess_input(x)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    msg = None
    if preds == 0:
        preds = "The Acne type is Blackheads"
        msg = 'Good to use products include Retinoids, Salicylic acid (Bea hydroxy acid- BHA), Benzoyl, peroxide, ' \
              'Lactic acid, Charcoal'
    elif preds == 1:
        preds = "The Acne type is Milia"
        msg = 'Good to use products include retinol or peptides (Option for mineral oil, petroleum), Retinoids, ' \
              'exfoliator, hydroxy acids (when use toner) Avoid using products including mineral oil, petroleum'
    elif preds == 2:
        preds = "Clear"
    elif preds == 3:
        preds = "The Acne type is Rosacea"
        msg = "Good to use products include Azelaic Acid, Niacinamide, Tranexamic Acide, Glycyrrhetinic Acid, " \
              "Centella Asiatica, Antioxidants"
    elif preds == 4:
        preds = "The Acne type is Scar"
        msg = 'Good to use products include Glycolic acid, Trichloracetic acid, Lactic accid, alpha-hydroxy acid, ' \
              'beta-hydroxy acid, salicylic acid'
    elif preds == 5:
        preds = "The Acne type is Tineafasialis"
        msg = "Good to use Anti-fungal cream or ointment that contains miconazole, clotrimazole, or terbinafine, " \
              "Shampoo that contains selenium sulfide Stop using skin care products that are oily. Use products that " \
              "are oil-free. The label may also read 'non-comedogenic'"
    else:
        msg = 'No treatment required'

    return preds, msg


def model_predict_skin_face(img_path, model):
    img_bytes = BytesIO(img_path)
    img = tf.keras.preprocessing.image.load_img(img_bytes, target_size=(224, 224))

    # Preprocessing the image
    x = tf.keras.preprocessing.image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 225
    x = np.expand_dims(x, axis=0)

    # x = preprocess_input(x)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    msg = None
    if preds == 0:
        preds = "The Skin type is Dry"
        msg = 'Good to use products include lanolin, shea butter, or waxes Avoid using products including glycolic ' \
              'acid, alpha-hydroxy acids, salicylic acid, and retinoic acid'
    elif preds == 1:
        preds = "The Skin type is Normal"
        msg = 'No treatment required'
    elif preds == 2:
        preds = "The Skin type is Oily"
        msg = 'Good to use products include Dimethicone, lactic, glycolic, and salicylic acid Avoid using products ' \
              'including paraffin, cocoa butter, or oils'

    return preds, msg

class ImageView(generics.CreateAPIView):
    serializer_class = ImageSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        image = serializer.validated_data['image']

        image = image.read()

        # Model saved with Keras model.save()
        MODEL_PATH = 'mainapp/model_acne_type.h5'

        # Load your trained model
        model = load_model(MODEL_PATH)

        acne_preds, msg1 = model_predict_acne_face(img_path=image, model=model)

        # Model saved with Keras model.save()
        MODEL_PATH2 = 'mainapp/model_skin_type.h5'

        # Load your trained model
        model2 = load_model(MODEL_PATH2)

        skin_preds, msg2 = model_predict_skin_face(img_path=image, model=model2)

        data = {
            'acne_preds': acne_preds,
            'acne_treatment': msg1,
            'skin_preds': skin_preds,
            'skin_treatment': msg2
        }

        return Response(data=data)

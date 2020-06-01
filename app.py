# import base64
# import json
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from PIL import Image
from io import BytesIO
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

import base64

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import expand_dims
from matplotlib import pyplot


app = Flask(__name__)
UPLOAD_FOLDER = "static/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "secret key"


def load_image(filename, size=(256, 256)):
    # load image with the preferred size
    pixels = load_img(filename, target_size=size)

    # convert to numpy array
    pixels = img_to_array(pixels)

    # scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5

    # reshape to 1 sample
    pixels = expand_dims(pixels, 0)

    return pixels


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    model = load_model("g_model_045600.h5", compile=False)

    name = request.form.get("name")
    input_img = request.files["input_img"]

    sname = secure_filename(input_img.filename)
    input_img.save(os.path.join(app.config["UPLOAD_FOLDER"], sname))

    # preprocess file
    image = load_image(input_img)

    # feed to saved model
    gen = model.predict(image)

    # scale generated image from [-1,1] to [0,1]
    gen_image = (gen + 1) / 2.0

    # save generated image to statis folder
    rname = "real" + "_" + input_img.filename
    filename2 = "./static/" + rname
    pyplot.imsave(filename2, gen_image[0])

    return render_template("gan.html", name=name, filename1=sname, filename2=rname)

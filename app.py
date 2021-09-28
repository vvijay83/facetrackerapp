from imutils import face_utils
import pandas as pd
import imutils
import dlib
import cv2
import numpy as np
import os
import boto3
from flask import Flask, render_template, request, send_file, jsonify
import time
from werkzeug.utils import secure_filename

s3 = boto3.client('s3', aws_access_key_id='',
         aws_secret_access_key= '', region_name='us-east-1')

bucket_name="facetracker-output"

app = Flask(__name__)

UPLOAD_FOLDER = 'data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home_page():
    return render_template('index.html')


def upload_file(file_name, bucket_name):
    """
    Function to upload a file to an S3 bucket
    """
    object_name = file_name
    response = s3.upload_file(file_name, bucket_name, object_name)
    return response

@app.route('/pointsraw', methods=['POST'])
def pointsraw():
    file = request.files['image']
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    return points(filename)

@app.route('/images', methods=['POST'])
def points_detection():
    file = request.files['image']
    filename = os.path.join(app.config['UPLOAD_FOLDER'], image_filename(file))
    file.save(filename)
    return points_images(filename)

def image_filename(filename):
    return filename.filename + '_processed.jpg'

def points_images(filename):
    def facepointer(image):
        rects = detector(image, 1)
        if len(rects) <1 :
            return pd.DataFrame({"A":[]})
        else:
        # loop over the face detections
            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region
                shape = predictor(image, rect)
                shape = face_utils.shape_to_np(shape)
            return(pd.DataFrame(shape))

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
    image = cv2.imread(filename, 0)
    image_col = cv2.imread(filename)

    df = facepointer(image)
    if df.empty:
        return jsonify({"status": "500", "message": "No face is able to be detected"})

    for i in range(0,68):
        try:
            image_col[df[1][i],df[0][i]-5:df[0][i]+5,2] = 0
            image_col[df[1][i] - 5:df[1][i] + 5, df[0][i], 2] = 0
            image_col[df[1][i], df[0][i] - 5:df[0][i] + 5, 1] = 0
            image_col[df[1][i] - 5:df[1][i] + 5, df[0][i], 1] = 0
        except:
            1

    cv2.imwrite(os.path.join(os.path.curdir, 'temp.jpg'), image_col)
    local_image = open('temp.jpg', 'rb')
    s3.put_object(Bucket=bucket_name, Key = filename, Body=local_image, ContentType= 'image/png')
    return send_file(os.path.join(os.path.curdir, 'temp.jpg'), as_attachment=True)


def points(filename):

    def facepointer(image):
        rects = detector(image, 1)
        if len(rects) <1 :
            return pd.DataFrame({"A":[]})
        else:
            # loop over the face detections
            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region
                shape = predictor(image, rect)
                shape = face_utils.shape_to_np(shape)
            return(pd.DataFrame(shape))

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
    image = cv2.imread(filename, 0)

    df = facepointer(image)
    if df.empty:
        return jsonify({"status": "500", "message": "No face is able to be detected"})

    output = {}
    for i in range(0,68):
        output[str(i) + '_vertical'] = str(df[0][i])
        output[str(i) + '_horizontal'] = str(df[1][i])

    return jsonify(output)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

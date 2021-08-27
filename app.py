#coding:utf-8
from imutils import face_utils
import pandas as pd
import imutils
import dlib
import cv2
import numpy as np
import os
from flask import Flask, render_template, request, send_file, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.curdir, 'data')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/density', methods=['POST'])
def wrinkle_density():
    file = request.files['image']

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    return winkle_density(filename)

@app.route('/images', methods=['POST'])
def wrinkle_detection():
    file = request.files['image']
    filename = os.path.join(app.config['UPLOAD_FOLDER'], image_filename(file))
    file.save(filename)
    return winkle_images(filename)

def image_filename(filename):
    return filename.filename + 'test.jpg'

def winkle_images(filename):

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
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
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

    cv2.imwrite(os.path.join(os.path.curdir, 'data', 'temp.jpg'), image_col)
    return send_file(os.path.join(os.path.curdir, 'data', 'temp.jpg'), as_attachment=True)

    rotation_correction_angle = -(df[0][27] / df[0][30] - 1) * 100
    image2 = imutils.rotate_bound(image, rotation_correction_angle)
    image_col = imutils.rotate_bound(image_col, rotation_correction_angle)
    df2 = facepointer(image2)
    if df2.empty:
        return jsonify({"status": "500", "message": "No face is able to be detected"})

    #Blur eyes, mouth and nose to remove non-wrinkle ridges
    facelength = df[1][8] - df[1][19]


    adaptive_thresh = cv2.adaptiveThreshold(image2,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    image4 = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2RGB)


    #blending out the right eye
    #typical_skin_colour = int(np.mean(image2[df2.iloc[31][1]:(df2.iloc[31][1] + 1), int((df2.iloc[31][0])*0.90):df2.iloc[31][0]]))
    right_eye_y = int(df2.iloc[36:42][0].mean())
    right_eye_x = int(df2.iloc[36:42][1].mean())
    right_eye_y_len = int((df2.iloc[36:42][0].max() - df2.iloc[36:42][0].min())/1.5)
    right_eye_x_len = int((df2.iloc[36:42][1].max() - df2.iloc[36:42][1].min()))
    masked_face = cv2.ellipse(image4.copy(),(right_eye_y ,right_eye_x ),(right_eye_y_len,right_eye_x_len),0,0,360,[255,255,255],-1)

    #blending out the left eye
    left_eye_y = int(df2.iloc[42:47][0].mean())
    left_eye_x = int(df2.iloc[42:47][1].mean())
    left_eye_y_len = int((df2.iloc[42:47][0].max() - df2.iloc[42:47][0].min())/1.5)
    left_eye_x_len = int((df2.iloc[42:47][1].max() - df2.iloc[42:47][1].min()))
    masked_face = cv2.ellipse(masked_face,(left_eye_y ,left_eye_x ),(left_eye_y_len,left_eye_x_len),0,0,360,[255,255,255],-1)


    #cutting out the forehead
    forehead_mask = np.zeros(masked_face.shape, dtype=np.uint8)
    forehead = [[df2.iloc[17][0], min(df2[1])], [df2.iloc[26][0], min(df2[1])], [int((df2.iloc[17][0] +  3 * df2.iloc[26][0])/4), min(df2[1]) - facelength/4], [int((3 * df2.iloc[17][0] + df2.iloc[26][0])/4), min(df2[1]) - facelength/4]]
    a3 = np.array( [forehead], dtype=np.int32 )
    forehead_mask = cv2.fillPoly( forehead_mask, a3, [255,255,255] )
    masked_forehead = cv2.bitwise_and(255-masked_face.copy(), forehead_mask)


    #cutting out the relevant pieces of the face
    mask = np.zeros(masked_face.shape, dtype=np.uint8)
    below_cheek = (df2.iloc[0:17].values)
    a3 = np.array( [below_cheek.tolist()], dtype=np.int32 )

    #shifting the cut-out 2 percent higher to exclude the chinbone lines from the image
    below_cheek = [[int(x[0]), x[1]*0.98] for x in below_cheek]
    #Cutting off the sides of the face to remove contours
    below_cheek[0][0] = below_cheek[0][0] * 1.05
    below_cheek[1][0] = below_cheek[1][0] * 1.05
    below_cheek[2][0] = below_cheek[2][0] * 1.05
    below_cheek[3][0] = below_cheek[3][0] * 1.025
    below_cheek[4][0] = below_cheek[4][0] * 1.025

    below_cheek[12][0] = below_cheek[12][0] * 0.95
    below_cheek[13][0] = below_cheek[13][0] * 0.95
    below_cheek[14][0] = below_cheek[14][0] * 0.95
    below_cheek[15][0] = below_cheek[15][0] * 0.975
    below_cheek[16][0] = below_cheek[16][0] * 0.975

    a3 = np.array( [below_cheek], dtype=np.int32 )
    mask = cv2.fillPoly( mask, a3, [255,255,255] )
    masked_face = cv2.bitwise_and(255-masked_face, mask)
    masked_face	= masked_face + (masked_forehead)


    #removing the nose
    nose_mask = np.zeros(masked_face.shape, dtype=np.uint8)
    nose = [(df2.iloc[27].values).tolist(), df2.iloc[31].values.tolist(), df2.iloc[32].values.tolist(), df2.iloc[33].values.tolist(), df2.iloc[34].values.tolist(), df2.iloc[35].values.tolist(), (df2.iloc[27].values).tolist()]
    #widening the nostrils
    nose[1][0] = nose[1][0]*0.9
    nose[5][0] = nose[5][0]*1.1
    a3 = np.array( [nose], dtype=np.int32 )
    nose_mask = cv2.fillPoly( nose_mask, a3, [255,255,255] )
    masked_face = cv2.bitwise_and(masked_face, 255-nose_mask)


    #removing the mouth
    mouth_mask = np.zeros(masked_face.shape, dtype=np.uint8)
    mouth = [(df2.iloc[48:60].values).tolist()]
    #widening the mouth
    mouth[0][0][0] = int(int(mouth[0][0][0])*0.95)
    mouth[0][6][0] = int(int(mouth[0][6][0])*1.05)
    #stretching the mouth higher
    mouth[0][1][1] = int(int(mouth[0][1][1])*0.975)
    mouth[0][2][1] = int(int(mouth[0][2][1])*0.95)
    mouth[0][3][1] = int(int(mouth[0][3][1])*0.95)
    mouth[0][4][1] = int(int(mouth[0][4][1])*0.95)
    mouth[0][5][1] = int(int(mouth[0][5][1])*0.975)
    #stretching the mouth lower
    mouth[0][7][1] = int(int(mouth[0][7][1])*1.025)
    mouth[0][8][1] = int(int(mouth[0][8][1])*1.05)
    mouth[0][9][1] = int(int(mouth[0][9][1])*1.05)
    mouth[0][10][1] = int(int(mouth[0][10][1])*1.05)
    mouth[0][11][1] = int(int(mouth[0][11][1])*1.025)
    #a3 = np.array( mouth, dtype=np.int32 )
    a3 = np.array( [mouth], dtype=np.int32 )
    mouth_mask = cv2.fillPoly( mouth_mask, a3, [255,255,255] )
    masked_face = cv2.bitwise_and(masked_face, 255-mouth_mask)

    masked_face = 255-masked_face


    median = cv2.medianBlur(masked_face,5)

    #Cutting off the empty spaces next to the face
    nonzero_columns = np.sum(median ==0, axis=0) > 0
    masked_face_cut = median.copy()[:,nonzero_columns[:,1], :]
    nonzero_rows = np.sum(median ==0, axis=1) > 0
    masked_face_cut = masked_face_cut[nonzero_rows[:,1],:, :]
    wrinkle_index = min(100, int(np.sum(masked_face_cut == 0)* 1000 / (masked_face_cut.shape[0] * masked_face_cut.shape[1] * masked_face_cut.shape[2])))


    #Calculating facial area wrinkle scores
    wrinkle_segregator = median.copy()
    cheeck_wrinkles = wrinkle_segregator[int(right_eye_x*0.85): df2.iloc[33][1], min(df2[0]):df2.iloc[33][0], :]
    cheeck_wrinkles2 = wrinkle_segregator[int(left_eye_x*0.85): df2.iloc[33][1], df2.iloc[33][0]:max(df2[0]), :]
    cheeck_wrinkle_index_right = min(100, int(np.sum(cheeck_wrinkles == 0) * 1000 / (cheeck_wrinkles.shape[0] * cheeck_wrinkles.shape[1] * cheeck_wrinkles.shape[2])))
    cheeck_wrinkle_index_left = min(100, int(np.sum(cheeck_wrinkles2 == 0)  * 1000/ (cheeck_wrinkles2.shape[0] * cheeck_wrinkles2.shape[1] * cheeck_wrinkles2.shape[2])))

    forehead_wrinkles = wrinkle_segregator[int(min(df2[1])*0.75):min(df2[1]) , min(df2[0]):max(df2[0]), :]
    forehead_wrinkle_index = min(100, int(np.sum(forehead_wrinkles == 0)* 1000 / (forehead_wrinkles.shape[0] * forehead_wrinkles.shape[1] * forehead_wrinkles.shape[2]) ))

    eye_wrinkles = wrinkle_segregator[int(right_eye_x*0.85):int(right_eye_x*1.15) , int((df2.iloc[0][0])):int((df2.iloc[33][0])), :]
    eye_wrinkles2 = wrinkle_segregator[int(left_eye_x*0.85):int(left_eye_x*1.15) , int((df2.iloc[33][0])):int((df2.iloc[16][0])), :]
    eye_wrinkle_index_right = min(100, int(np.sum(eye_wrinkles == 0) * 1000/ (eye_wrinkles.shape[0] * eye_wrinkles.shape[1] * eye_wrinkles.shape[2]) ))
    eye_wrinkle_index_left = min(100, int(np.sum(eye_wrinkles2 == 0) * 1000 / (eye_wrinkles2.shape[0] * eye_wrinkles2.shape[1] * eye_wrinkles2.shape[2])))

    mouth_wrinkles = wrinkle_segregator[int(df2.iloc[33][1]):int(max(df2[1])) , int(min(df2[0])):int(max(df2[0])), :]
    mouth_wrinkle_index = min(100, int(np.sum(mouth_wrinkles == 0) * 1333/ (mouth_wrinkles.shape[0] * mouth_wrinkles.shape[1] * mouth_wrinkles.shape[2]) ))

    median[:,:,0] = image4[:,:,0]
    median = 255 - median


    filename_s = "{}_{}_{}_{}_{}_{}_{}".format(wrinkle_index,forehead_wrinkle_index,eye_wrinkle_index_right,eye_wrinkle_index_left,cheeck_wrinkle_index_right,cheeck_wrinkle_index_left,mouth_wrinkle_index)
    cv2.imwrite(os.path.join(os.path.curdir, 'data', filename_s + 'w.jpg'), median)
    return send_file(os.path.join(os.path.curdir, 'data', filename_s + 'w.jpg'), as_attachment=True)



def winkle_density(filename):

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
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
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
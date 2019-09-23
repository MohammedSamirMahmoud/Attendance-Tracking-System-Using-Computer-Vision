from .Variables import EMBEDDINGS_PATH, SVM_MODEL_PATH,SVM_CONFIDENCE, model
from .Get_Embeddings import get_embedding
import pickle
import cv2
import numpy as np
from numpy import asarray
from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from PIL import Image
import mtcnn
from mtcnn.mtcnn import MTCNN


detector = MTCNN() # Creating instance from the class MTCNN

def multiple_faces(filename, required_size=(160, 160)):
  data = load( EMBEDDINGS_PATH + '/Embeddings-dataset.npz')
  trainy = data['arr_1']
  out_encoder = LabelEncoder()
  out_encoder.fit(trainy)
  trainy = out_encoder.transform(trainy)
  
  # LOAD THE MODEL
  print("TESTING ON AN IMAGE")
  print("LOADING THE MODEL...")
  svm_model = pickle.load(open(SVM_MODEL_PATH +  '/svm_model.sav', 'rb'))
  print("DONE LOADING THE MODEL!")
  print("LOADING THE IMAGE...")
  image = Image.open(filename) # load image from file
  print("DONE LOADING THE IMAGE!")
  image = image.convert('RGB') # convert to RGB, if needed
  pixels = asarray(image) # convert to array
  results = detector.detect_faces(pixels) # detect faces in the image
  
  if(len(results) == 0):
    return False
  
  
  # LOOP OVER ALL FOUND FACE AND ANNOTATE THEM
  for i in range(len(results)):
    
    x1, y1, width, height = results[i]['box'] # extract the bounding box from the i-th first face
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2] # extract the face
    image = Image.fromarray(face)
    image = image.resize(required_size) # resize pixels to the model size
    face_array = asarray(image)
    face_emb = get_embedding(model, face_array) # get the embeddings
    samples = expand_dims(face_emb, axis=0)
    prediction = svm_model.predict(samples)
    predict_name = out_encoder.inverse_transform(prediction)
    pred_proba = svm_model.predict_proba(samples)
    
    # predict_name = out_encoder.inverse_transform(prediction)
    # print("Prediction is",predict_name ,prediction,pred_proba )
    
    print(pred_proba[0][prediction[0]])
    if(pred_proba[0][prediction[0]] >= SVM_CONFIDENCE): # SVM Thresholding to Get Known vs UnKnown People was 0.999
      predict_name = out_encoder.inverse_transform(prediction)
      print("Prediction is",predict_name)
    else:
      predict_name = ['UNKNOWN']
      print("Prediction is",predict_name)

    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(pixels, predict_name[0], (x1, y1), font, 0.7, (255, 255, 0), 2)
  
  print("SAVING OUTPUT IMAGE...")
  cv2.imwrite( filename , pixels) 
  print("IMAGE SAVED TO:" , 'D:\\Work\\Attendance\\media\\testing\\IMG_1607.JPG' )
  return filename
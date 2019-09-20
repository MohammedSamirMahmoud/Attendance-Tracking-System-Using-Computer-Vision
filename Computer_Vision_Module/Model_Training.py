from .Load_Dataset import load_dataset
from .Get_Embeddings import get_embedding
from .SVM_Classifier import svm_train
from .Variables import model,TRAINING_IMAGES_PATH,EMBEDDINGS_PATH
import sys
import os
from os import listdir
from os.path import isdir
import numpy as np
from numpy import asarray
from numpy import savez_compressed
from numpy import load
import keras




def trainModel():
  # load train dataset
  print("STARTING THE TRAINING PROCESS...")
  print("DATA PATH IS:",TRAINING_IMAGES_PATH)
  
  if not isdir(TRAINING_IMAGES_PATH):
    print("ERROR: GIVEN PATH IS NOT A DIRECTORY, PATH IS:", TRAINING_IMAGES_PATH)
    print("TERMINATING...")
    sys.exit(0)
  trainX, trainy , flag = load_dataset(TRAINING_IMAGES_PATH + '/')
  print("SHAPE OF TRAINING EXAMPLES:",trainX.shape,"SHAPE OF LABELS:" ,trainy.shape)
  
  # CONVERT EACH FACE IN THE TRAIN SET TO AN EMBEDDING
  newTrainX = list()
  for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
  newTrainX = asarray(newTrainX)
  print("SHAPE OF TRAINING EMBEDDINGS:",newTrainX.shape,"SHAPE OF LABELS:" ,trainy.shape)
  # If true save if false append?
  print("SAVING THE EMBEDDINGS NUMBY ARRAY...")
  if flag == True:
    # Data base is empty, save new numpy array and keep newTrainX
    savez_compressed( EMBEDDINGS_PATH + '/Embeddings-dataset.npz', newTrainX,trainy)  # save arrays to one file in compressed format WILL BE USED WHEN WE ADD NEW PERSON TO THE DATABASE WE'LL LOAD THIS AND APPEND
    print("STARTING TRAINING THE SVM MODEL...")
    svm_train(newTrainX, trainy)
  elif flag == False:
    data = load( EMBEDDINGS_PATH + '/Embeddings-dataset.npz' )
    old_embeddings , old_labels = data['arr_0'] , data['arr_1']
    print("OLD EMBEDDINGS:", old_embeddings.shape ,"OLD LABELS:" ,old_labels.shape)
    new_embeddings , new_labels = newTrainX , trainy
    print("NEW EMBEDDINGS:", new_embeddings.shape ,"NEW LABELS:" ,new_labels.shape)
    trainX  = np.concatenate((old_embeddings, new_embeddings), axis=0)
    trainy  = np.concatenate((old_labels, new_labels), axis=0)
    print("FINAL EMBEDDINGS:", trainX.shape ,"FINAL LABELS:" ,trainy.shape)
    savez_compressed( EMBEDDINGS_PATH + '/Embeddings-dataset.npz' , trainX,trainy) 
    svm_train(trainX, trainy)
  
import numpy as np
from numpy import load
from numpy import asarray
from numpy import savez_compressed
from .Variables import EMBEDDINGS_PATH,model,PATH_TO_ADDED_PERSON_FOLDER
from .Load_Dataset import load_dataset
from .Get_Embeddings import get_embedding
from .SVM_Classifier import svm_train

def add_person():
  '''
  Gets Images Embeddings of new person and concatenate them with the old Embeddings
  '''

  # load train dataset --> Returns NewTrainx & NewTrainY
  #print("Data is at:",PATH_TO_ADDED_PERSON_FOLDER)
  trainX_new_person, trainy_new_person ,_ = load_dataset(PATH_TO_ADDED_PERSON_FOLDER + '/')
  #print("SHAPE OF ADDED TRAINING EXAMPLES:",trainX.shape,"SHAPE OF LABELS:" ,trainy_new_person.shape)
  
  # CONVERT EACH FACE IN THE TRAIN SET TO AN EMBEDDING
  newTrainX_Images = list()
  for face_pixels in trainX_new_person:
    embedding = get_embedding(model, face_pixels)
    newTrainX_Images.append(embedding)
  newTrainX_Images = asarray(newTrainX_Images)
  print("SHAPE OF ADDED EMBEDDINGS:",newTrainX_Images.shape,"SHAPE OF LABELS:" ,trainy_new_person.shape)


  data = load( EMBEDDINGS_PATH + '/Embeddings-dataset.npz' )
  old_embeddings , old_labels = data['arr_0'] , data['arr_1']
  print("OLD EMBEDDINGS:", old_embeddings.shape ,"OLD LABELS:" ,old_labels.shape)
  new_embeddings , new_labels = newTrainX_Images,trainy_new_person
  print("NEW EMBEDDINGS:", new_embeddings.shape ,"NEW LABELS:" ,new_labels.shape)
  trainX  = np.concatenate((old_embeddings, new_embeddings), axis=0)
  trainy_all_persons  = np.concatenate((old_labels, new_labels), axis=0)
  print("FINAL EMBEDDINGS:", trainX.shape ,"FINAL LABELS:" ,trainy_all_persons.shape)
  savez_compressed( EMBEDDINGS_PATH + '/Embeddings-dataset.npz' , trainX,trainy_all_persons) 
  svm_train(trainX, trainy_all_persons)
  return
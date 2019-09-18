from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
import os
from .Variables import SVM_MODEL_PATH


def svm_train(newTrainX, trainy ):  
  in_encoder = Normalizer(norm='l2')
  newTrainX = in_encoder.transform(newTrainX)
  out_encoder = LabelEncoder() # label encode targets
  out_encoder.fit(trainy)
  trainy = out_encoder.transform(trainy)
  svm_model = SVC(kernel='linear', probability=True)
  svm_model.fit(newTrainX, trainy)
  print("SUCCESSFULLY FINISHED TRAINING THE SVM MODEL SUCCES!")
  yhat_train = svm_model.predict(newTrainX)
  score_train = accuracy_score(trainy, yhat_train) # score
  print('ACCURACY: TRAIN=%.3f' % (score_train*100))
  # SAVING THE MODEL
  print("SAVING SVM MODEL TO:", SVM_MODEL_PATH +  '/svm_model.sav ...')
  filename = SVM_MODEL_PATH +  '/svm_model.sav'
  pickle.dump(svm_model, open(filename, 'wb'))
  print("DONE SAVING THE MODEL!")
  return
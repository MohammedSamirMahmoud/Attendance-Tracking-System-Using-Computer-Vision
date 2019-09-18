from .Load_Faces import load_faces
from .Variables import EMPLOYEES_NAMES,EMBEDDINGS_PATH

import os
import numpy as np
from numpy import savez_compressed
from numpy import asarray
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from numpy import load
from os.path import isdir
from os import listdir



# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
  '''
  THIS FUNCTION TAKE DATABASE DIRECTORY WITH PERSONS' NAMES AS SUBDIRECTORY
  RETURNS LABELS AND NUMPY ARRAYS
  '''
  flag = None
  X, y = list(), list()
  # savez_compressed( EMBEDDINGS_PATH + '/Embeddings-dataset.npz', newTrainX,trainy)  # save arrays to one file in compressed format WILL BE USED WHEN WE ADD NEW PERSON TO THE DATABASE WE'LL LOAD THIS AND APPEND
  # data = load( EMBEDDINGS_PATH + '/Embeddings-dataset.npz')
  # trainy = data['arr_1']
  
  if os.path.isfile( EMPLOYEES_NAMES +'/Employees-dataset.npz') == True:
    data = load( EMPLOYEES_NAMES + '/Employees-dataset.npz')
    Employees = data['arr_0'] # List
    # Already exist
    flag = False
  else:
    Employees = np.array([])
    flag = True
  
  print("CURRENT EMPLOYEE DATABASE:" ,Employees )
    
  
  for subdir in listdir(directory): # LOOP OVER SUB-FOLDERS, on per class
    if subdir in Employees:
      print("SKIPPING EMPLOYEE" , subdir)
      continue
    Employees = np.append(Employees , subdir)
    path = directory + subdir + '/' # PATH = SUBFOLDER
    if not isdir(path): # skip any files that might be in the dir
      print("WARNING: FILES EXIST IN THE DATA DIRECTORY (ONLY FOLDERS ARE READ)!")
      print("SKIPPING FILE" , path , "...")
      continue
    
    faces = load_faces(path) # load all faces in the subdirectory
    labels = [subdir for _ in range(len(faces))] # create labels FOR THE PERSON, WE DO RANGE BECAUSE DETECTOR CAN MISS SOME IMAGES AND DETECT 0 FACES
    print('   >LOADED %d EXAMPLES FOR CLASS: %s' % (len(faces), subdir)) # summarize progress
    X.extend(faces)  # store
    y.extend(labels) # store
    savez_compressed( EMPLOYEES_NAMES + '/Employees-dataset.npz', Employees )  # save arrays to one file in compressed format WILL BE USED WHEN WE ADD NEW PERSON TO THE DATABASE WE'LL LOAD THIS AND APPEND
  return asarray(X), asarray(y) , flag
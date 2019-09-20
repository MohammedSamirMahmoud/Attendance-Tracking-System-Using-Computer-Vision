import os
import mtcnn
from mtcnn.mtcnn import MTCNN

detector = MTCNN() # Creating instance from the class MTCNN

# EMBEDDINGS PATH
EMBEDDINGS_PATH = os.getcwd() 

# EMPLOYEES PATH
EMPLOYEES_NAMES = os.getcwd()

# SVM Model Path
SVM_MODEL_PATH = os.getcwd()

# Create Folders/Classes under photos dir in your Django Project
TRAINING_IMAGES_PATH = os.getcwd() + '\\..\\media\\photos'


# Path of facenet_keras.h5 file
PATH_TO_FACENET_MODEL = os.getcwd()
FACENET_MODEL = PATH_TO_FACENET_MODEL + '\\facenet_keras.h5'

if os.path.isfile(FACENET_MODEL) == True:
  print("facenet_keras.h5 already exist, no need to re-download it \n")
else:
  print("Working directory where detector model will be saved in:",FACENET_MODEL)
  bashCommand = "wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PZ_6Zsy1Vb0s0JmjEmVd8FS99zoMCiN1' -O " + FACENET_MODEL
  os.system(bashCommand)

model = load_model(FACENET_MODEL)
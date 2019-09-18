import os
import mtcnn
from mtcnn.mtcnn import MTCNN

detector = MTCNN() # Creating instance from the class MTCNN

# EMBEDDINGS PATH
EMBEDDINGS_PATH = os.getcwd() 

# EMPLOYEES PATH
EMPLOYEES_NAMES = os.getcwd()
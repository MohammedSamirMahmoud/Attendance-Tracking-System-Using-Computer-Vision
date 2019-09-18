import os
import numpy as np
from os import listdir
from numpy import asarray
from numpy import load


from .Extract_Faces import extract_face


# load images and extract faces for all images in a directory
def load_faces(directory):
  '''
  THIS FUNCTION TAKES DIRECTORY OF (ONE) PERSON IMAGES AS INPUT
  OUTPUTS A FACE NUMPY ARRAY
  '''
  faces = list()
  # enumerate files
  for filename in listdir(directory):
    path = directory + filename
    face = extract_face(path) # get face
    if(len(face) == 0):
      continue
    faces.append(face) # store
  return faces
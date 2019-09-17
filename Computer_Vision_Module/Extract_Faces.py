from PIL import Image
import mtcnn
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from numpy import load
from .Variables import detector


def extract_face(filename, required_size=(160, 160)):
  '''
  THIS FUNCTION TAKES (ONE) IMAGE AS INPUT
  DETECT AND RETURN (ONE) FACE ARRAY RESIZED --> Using the MTCNN Detector
  '''
  image = Image.open(filename) # load image from file
  image = image.convert('RGB') # convert to RGB, if needed
  pixels = asarray(image) # convert to array
  results = detector.detect_faces(pixels) # detect faces in the image
  if(len(results) == 0):
    return []
  x1, y1, width, height = results[0]['box'] # extract the bounding box from the first face
  x1, y1 = abs(x1), abs(y1)
  x2, y2 = x1 + width, y1 + height
  face = pixels[y1:y2, x1:x2] # extract the face
  image = Image.fromarray(face)
  image = image.resize(required_size) # resize pixels to the model size
  face_array = asarray(image)
  return face_array
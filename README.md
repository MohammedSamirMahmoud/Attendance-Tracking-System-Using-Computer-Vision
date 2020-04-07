# Attendance-Tracking-System-Using-Computer-Vision

![Alt Text](https://www.olloltd.com/images/2020/02/sssss.gif)
## Introduction:
Tracking Systems nowadays are in high demand starting from the very basic attendance systems to most critical security systems. With the powerful rise of AI & Computer Vision. These Tracking Systems have become more accurate & more precise resulting in more reliability and stability. 
Our Project is basically Leveraging The power of Image & Video Detection, Recognition & Tracking. 
We have applied the best-in-class Computer Vision Algorithms to Build a full Attendance Tracking System Enabling the following Features: 
- Real Time Face Detection & Recognition applied on Video Camera Streams. “Based on your    Computational Resources Power”
- Ability to Add new Employees to the System and Retrain the Algorithm Using Only 1~2 Images. 
- A Web Application Showing Employees Attendance & Activities.
- Extras: Faces Detector & Recognizer

**Final Target** : Monitor How Much Time Does an employee stays in the Work Environment vs how much 
does he/she waste per day either outside the environment or in fun area.

## System Architecture Overview:
Let’s define it in simple forms, Any System has its Own Input Data, Output Data, Processing Unit, Storage Unit.
INPUT 🡪 Data Source for this project is mainly live streams (Videos) from environment cameras.
Desired (Output) 🡪 A Reliable Employees Attendance/Activity DATABASE on a Web Application.
In Between (Target) 🡪 Build an Attendance Tracking System Using Computer Vision.
Storage 🡪 Videos HDD + Store Employees Final Activity in a MySQL Database.
![image](https://drive.google.com/uc?export=view&id=139tuRPWjPuxKi8LcYqPIxHKq7wDUDZzt)
Now that we have seen the overall System Overview, Let's Build the Pipeline for this Application.
## System Pipeline:
The Project will mainly be separated into 4 modules:
Module A: Getting Data in Either from Videos or From Images
Module B: Computer Vision Part (Face Detector & Face Recognizer)
Module C: Attendance Logic (Based on Your Environment) 
Module D: Web Application (Attendance Admin Panel & Testing Interface) 
![image](https://drive.google.com/uc?export=view&id=15D50d7XtIne48o0_eI_o4EFJmcqlgHgQ)
Let’s dig into each module in a brief:

## Module A: Getting Data in Either from Videos or From Images
Since this is an attendance Tracking System then we have some cameras installed at some location in your environment
![image](https://drive.google.com/uc?export=view&id=1B9AHRKxJQ0hJB6Pr2rkER9q0XOD1Fk70)
We have Installed 6 Cameras at the following Critical Locations:
    A – Entrance Door from Outside (Camera 3 & 4) 🡪 To get Employees leaving Environment.
    B – Entrance Door from Inside (Camera 1 & 2) 🡪 To get Employees Coming into The Environment.
    C – Fun Area Door (Camera 5 & 6) 🡪 To Capture Employees Going in or Out of Fun Area.
You may ask Why at these locations: It’s as simple as we want to know whenever an employee arrives to the Environment using cameras 1 & 2 and whenever he/she leaves using cameras 3 &4 . And how much time does he/she wastes at Fun Area.
N.B : We have used 2 cameras at each zone to cover all possible face angels.
We need for Each Video Frame to detect faces inside each frame & Recognize it. (You may think of a frame as an image).

## Module B: Computer Vision Part (Face Detector + Face Recognizer)
Here we want to do 2 main tasks first detecting if an image contains a face or not (Detection). then if the image has a face we want to know who is this person (Recognition).
### Module B.1 Face Detection
At this step we have some Input Videos from different Cameras. Videos are composed of Frames, you could think of each frame as a     
normal image. And we just want to detect if there is a face in an image or not.
### Detection Trials:
#### B.1.1 HAAR
We have started our trials using traditional image processing algorithms “HAAR” it is a feature based cascade classifier using some basic filter we can get:
-The region of the eyes is often darker than the region of the nose and cheeks.
-The eyes are darker than the bridge of the nose
![image](https://drive.google.com/uc?export=view&id=1uV2H8A2GXLl_23yFJ5Flz0NjoYS8y_ZO)
Since that HAAR is very inefficient in difficult lighting conditions we didn’t expect a good output from it. Also because Haar Features have to be determined manually, there is a certain limit to the types of things it can detect. If you give a classifier (a network, or any algorithm that detects faces) edge and line features, then it will only be able to detect objects with clear edges and lines. Even as a face detector, if we manipulate the face a bit (say, cover up the eyes with sunglasses, or tilt the head to a side), a Haar-based classifier may not be able to recognize the face.
However HAAR Has its own Bright Side when it’s applied on high quality clear images: Since that we don’t need to train HAAR Features as they are manually selected we just need a relatively small amount of Data for faces also HAAR execution speed is high and requires less computational resources when compared to other detection algorithms.
HAAR Output on Our Environment: 
![image](https://drive.google.com/uc?export=view&id=1WyHh8uVqDhezY-eoh4CXUE-N0U0n6Lo_)		 
As you can see HAAR failed to get our faces in this poor position & lightning however it detected basic edges in the image.

#### B.1.2 YOLO (You Only Look Once)
YOLO is an object detector which detects objects in both images and video streams using Deep Learning, OpenCV, and Python. We call YOLO as single stage detector ,there are other multi stage detectors that are more accurate but are very slow.
The “You Only Look Once,” or YOLO, family of models are a series of end-to-end deep learning models designed for fast object detection, developed by Joseph Redmon, et al. and first described in the 2015 paper titled “ You Only Look Once: Unified, Real-Time Object Detection” , details an object detector capable of super real-time object detection, obtaining 45 FPS on a GPU.
![image](https://drive.google.com/uc?export=view&id=1_0gRQtEZ-wMSR7H2w4aOIkNQympjN9GD) 
YOLO Detection Method: Sliding Window
This approach involves a single deep convolutional neural network (originally a version of GoogLeNet, later updated and called DarkNet based on VGG) that splits the input into a grid of cells and each cell directly predicts a bounding box and object classification “Called Sliding Window”. The result is a large number of candidate bounding boxes that are consolidated into a final prediction by a post-processing step. Since YOLO is an Object Detector then it will be able to detect different objects in the image, including people.

![image](https://drive.google.com/uc?export=view&id=1cpmIA-x5QLOZQiUArsyqgSIy_eIkOHAb)

Perfect Object (Human) Detection isn’t it ?! 
Simple answer is --> Yes it’s very good. But only in Full Object Detection. I mean the output of the detector is a box containing the person.
![image](https://drive.google.com/uc?export=view&id=1pCDALqHllSpHZhOai42u3NsuCxPd-GtH)   
When we feed these images to a classifier (Person A “left” & Person B “Right”), a classifier will always look at unique easy features at the beginning, this includes the background out there ,T-shirts Colors,etc which is not a good thing.
I.e After some training whenever we pass to the classifier an image of Any person standing for example at the location that has green background where Person_B Stands “Right Image” ,the classifier will always predicts Whoever passes by this location as Person_B because it fires based on the background not actual features.

#### B.1.3 Segmentation
Now we have seen that YOLO is a good detector , but the detection outputs a full box including a background which we want to eliminate. The Segmentation Concept Comes into play here.
Segmentation will allow us to crop the Region of Interest (ROI) that we want from the image i.e eliminating the whole background.
![image](https://drive.google.com/uc?export=view&id=1U0VRZm2bqkEwHpM-lGOIMszd4xT1t8mV)

This Result is relatively perfect but actually we don’t want neither the background nor the whole body. And the detector still getting the whole body.
We just want the faces! That's why we have checked for other face detectors and we found something called YOLO FACES.

#### B.1.4 YOLO FACES
This is a Special Version of YOLO Object Detectors that focuses only on detecting faces as an objects rather than detecting the whole body.
We have tried it however it wasn’t very good in detecting all faces whenever they are aligned to the right or the left.

![image](https://drive.google.com/uc?export=view&id=1cLS2x4xN8OIQHoFROgbVeGFwMZztWG4i)

Finally we have found a much better detector than most of the previously mentioned algorithms.

#### B.1.5 MTCNN
Multitask Cascaded Convolutional Neural Networks (MTCNN), The MTCNN is popular because it achieved then state-of-the-art results on a range of benchmark datasets, and because it is capable of also recognizing other facial features such as eyes and mouth, called landmark detection.

![image](https://drive.google.com/uc?export=view&id=1xNF6GMGnj06reV4iZBGcxmYmgGh6hLRx)  
Obama Face Landmarks


### Detection Summary:

     Detection Algorithm	    Detection Accuracy
           HAAR	                    70 %
           YOLO	                    92 %
        Segmentation	              90 %
         YOLO Faces	                80 %
           MTCNN	                  96 %

### Module B.2 Face Recognition
At this step we have some Images that contain boxes around any face in the picture. Now it’s time to recognize the faces and know who is the exact person appearing.
This could be considered as a normal image classification problem.
Where we can use either traditional ML Algorithms or Deep Learning ones. We have tried the very basic ML Algorithms to the most advanced ones & Mixing between them. Here is our Trials
    -	Logistic Regression & Random Forests (Very Poor Accuracy).
    -	Neural Networks (Poor Accuracy on Images).
    -	Convolutional Neural Networks “CNN” (Excellent Accuracy on Images, However Since this is an Attendance System hence we have only a       few registered images for each employee “1~4 Images” which aren’t enough at all to train a CNN From Scratch).
    -	Siamese Networks (Based on 2 CNNs) → Good one with decent accuracy = 84% .
    -	CNN + Deep NN → Using Transfer Learning we have used a pre trained CNN to Extract Face Features then we have applied the output 
      features (128 Embeddings) to a NN which resulted in an overall accuracy = 90% .
    -	CNN + SVM → Using Transfer Learning we have used a pre trained CNN to Extract Face Features then we have applied the output  
      features (128 Embeddings per Face) to SVM which resulted in mostly an overall system accuracy = 96% .
      
N.B You may think how come CNN + NN doesn’t achieve high accuracy like CNN+SVM, It’s simply because Deep learning requires alot of data to get good accuracy. In the best scenario we will have only 5 images per employee and for example 100 Employees.  (i.e 100 x 5 x 128 Embeddings) which is not enough for any deep learning model.
Therefore we can conclude all of these by saying that the main pillars for the Recognition phase are CNNs , Transfer Learning & SVM..

#### Final Pipeline:
![image](https://drive.google.com/uc?export=view&id=1Fq8eNMrHuLMSXCyDYXkWKemVMT6f3lSo)

## Module C: Attendance Logic (Based on Your Environment)
This part is going to be purely SW Logic, Covering 2 Parts 
    -	Firstly, what about the 4% Error rate in our Classification, how can we make sure that the person recognized is 100% that person 
      not 96%. → Solved using Confidence Interval
    -	Secondly we need some logic to handle the first seen & last seen of a person including his/her wasted time in fun area or outside 
      the company & then calculate the final working & wasted hours .
Let's Start, Each of our cameras will produce a log file at the end of the day. the logic module combines and process on them.
![image](https://drive.google.com/uc?export=view&id=13zsinfQrZZaI5aHgP3JEh5TCE8E1T-Z4)
Sample log from camera_id 1

Before applying the confidence interval concept, firstly, we will combine all of the six log files produced by the 6 cameras and sort by:
      -	The employee name
      -	Timestamp 
      -	The Camera ID  
![image](https://drive.google.com/uc?export=view&id=1xEkVxnXt19e1afesKHM4jAwAfM7yvf07)
Then we unify the cameras Id’s and reduce them from 6 Ids to 3 Ids “Since each 2 cameras capture the same area but from different angles”. ( 1 & 2 → Reduced to 1  , 3 & 4 → Reduced to 3 ,  5 & 6 → Reduced to 6 )
![image](https://drive.google.com/uc?export=view&id=1bRaWL76osgX8igB8uz9bzthUhlPH13nB)
We have set our confidence interval threshold to be 5 , Which means if a person was detected in a video stream in 5 consecutive frames then most probably he/she is the same person. However if less than 5 then mostly due to light conditions or face angel the classifier model has mistaken the classification task. Therefore drop these log  entries from your log file.
Here is the final accurate log file:
![image](https://drive.google.com/uc?export=view&id=19E8SZ3HrUv_15VrOvg93LbokySGwiapJ)
Then by doing some pair successive subtractions, we can obtain the time an employee spent in each area “you can check the code at the GitHub repository for further investigation.
And here is the final log file to be imported in the web application database containing all employees activity data. “Note that we scaled the 8 of the working day to 8 minutes for computational reasons”
![image](https://drive.google.com/uc?export=view&id=1TNBTjdE3-n4kyzYy_44JXKmVPOuZ6Shb)
Now it’s time to package all of this work inside a web application.


## Module D: Web Application (Attendance Admin Panel)
Now we have reached the state of having a really good model without real life application using it, so it’s useless so far in our case we have built a web application that track a company’s employees activities/attendance, activities like when has  employee arrive workplace, how many hours the employee was off working, and how many hours the employee was on desk and different types of activities that the employees usually do during a  working day in the company’s workplace.

In the web application module we will go through how we have built our web application using Django Framework.
### Activity Tracking Web Application  will help us to

        1.	Add, Edit, Delete Employees to our Database.
        2.	Train our model for our newly added employees.
        3.	Test our model on a single picture.
        4.	Take a live snapshot and test the snapshot against our machine learning model.
        
### Web Application Database Entities
Our web application will consist of Employees, Activities that occurred by employees and Employees Pictures and Test Image model, each of these models represents an Entity with following responsibility:

        A.	Employee: Responsible for adding, removing, editing employees.
        B.	Activity: Responsible for employees activities.
        C.	Picture: Responsible for adding picture to an employee.
        D.	Test: Responsible for testing an image against our machine learning model.

Each employee has a set of activities for each working day associated with him, each employee has a set of pictures that the machine learning model will use to train and test the employee’s face and a Test entity to be able to test an image or video against machine learning model through the web application.
Django comes with a grateful built-in admin panel that will help us to build our Web Application admin panel quickly, all we have to do is to register our models in the admin panel so the admin user will be able to make CRUD operations in all models.

### Django Project installation:
To install django on our machine it’s recommended to install it on Python virtual environment.
I will suppose that you have python3 and virtual environment on linux machine  already installed on your machine, so we will create new virtual environment as following
  1.	make a new directory that will represent our web project and application.
  
                      $ mkdir Attendance_Tracking_System_Using_Computer_Vision
  2.	Inside Attendance_Tracking_System_Using_Computer_Vision we will run the following to create our virtual environment with the name 
      venv
      
                      $ virtualenv ./venv
  3.	After creating our virtual environment we need to activate it, we will run the following command in the same directory.
      
                      $ source ./venv/bin/activate
  4.	After activating our virtual environment we will need to install python-pip, if you have pip already installed you don’t have to 
      reinstall it.
      
                      $ sudo apt install python3-pip
  5.	Now we need to install the requirements to get our web application working, you will find the requirements file in the Github 
      repository link below, after copying file to your project root directory run the below command to install packages.
      
                      $ pip install -r /path/to/requirements.txt

  6.	Now let’s create our new project that will host our web application, to create a new Django project just run the following 
      command, note the “./’ tell django to create a project in the same directory which should be the root directory to our web 
      application.
                      
                      $ django-admin createproject 
                      $ Attendance_Tracking_System_Using_Computer_Vision ./
                      
  7.	A django project consists of multiple applications each application should represent a component of the project components, so now       we need to create an application that will represent our activity tracking application on the Attendance Tracking System Project, 
      to create new application just run the following command.
                      
                      $ ./manage.py createapp Activity_Tracking_Web_App

  8.	After that you should have a specific folder for your machine learning function in the project root directory, so you will add all 
      your machine learning scripts in a subfolder named “Computer_Vision_Module”
  9.	Very good till now we have installed our project environment and initialize the project structure, now you should see a file 
      structure like the following in your project root directory.
      
                  │   .gitignore
                  │   manage.py
                  │   requirements.txt
                  ├───Activity_Tracking_Web_App
                  │   │   admin.py
                  │   │   apps.py
                  │   │   models.py
                  │   │   tests.py
                  │   │   views.py
                  │   │   __init__.py
                  │   │
                  │   ├───migrations
                  │   │   │   __init__.py
                  ├───Attendance_Tracking_System_Using_Computer_Vision
                  │   │   settings.py
                  │   │   urls.py
                  │   │   wsgi.py
                  │   │   __init__.py
                  └───Computer_Vision_Module
                          Extract_Faces.py
                          Load_Dataset.py
                          Load_Faces.py
                          Variables.py

  10.	Now we should run our local server to make sure that we have correctly set up our project, run the following command to run local 
      development server.
      
                      $ ./manage.py runserver
  The output should be
  
                    Django version 2.2.4, using settings 'Attendance_Tracking_System_Using_Computer_Vision.settings'
                    Starting development server at http://127.0.0.1:8000/
                    Quit the server with CTRL-BREAK.


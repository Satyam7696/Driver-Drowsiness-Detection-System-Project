import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time
import imutils


mixer.init()
sound = mixer.Sound('2.wav')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
model = load_model(os.path.join("models", "model1.h5"))

path = os.getcwd()

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
lbl=['Close', 'Open']
status = ""
color = (255, 255, 255)

cap = cv2.VideoCapture(0)
frame_counter = 0
frame_interval = 5  # Only process every fifth frame

while True:
    ret, frame = cap.read()
    frame_counter += 1
    if frame_counter % frame_interval == 0:
        height,width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,minNeighbors = 3,scaleFactor = 1.1,minSize=(25,25))
        eyes = eye_cascade.detectMultiScale(gray,minNeighbors = 2,scaleFactor = 1.1)

        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,0,0) , 3 )
            for (x,y,w,h) in eyes:
                eye = frame[y:y+h,x:x+w] 
                eye = cv2.resize(eye,(120,120))
                eye = eye/255
                eye = eye.reshape(120,120,3)
                eye = np.expand_dims(eye,axis=0)
                prediction = model.predict(eye)
                print("closed : ",prediction[0][0])
                print("Open : ",prediction[0][1])
               #Condition for Close
                if prediction[0][0]>0.40:
                    cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
                    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
                    score=score+1
                    #write on lED
                    
                    time.sleep(2)
                    status = "SLEEPING !!!"
                    color = (0, 0, 255)
                    if(score >=15 ):
                        try:
                            sound.play() 
                            
                        except:  # isplaying = False  
                            pass
                    if(score <15):
                        sound.stop()
                        
                #Condition for Open
                elif prediction[0][1] > 0.70:
                    score = score - 1
                    if (score < 0 and score < 5):
                        score = 0
                        sound.stop()
                    cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
                    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
                    time.sleep(2)
                    status = "Active :)"
                    color = (0, 0, 255)

        cv2.imshow('Driver drowsiness system',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sound.stop()
            break
cap.release()
cv2.destroyAllWindows()

import face_recognition
from PIL import Image
import numpy as np
import pickle
import cv2
import os

with open('pwd.txt', 'r') as pwd:
    folder_location = pwd.read()

def find_encodings(images_):
    encode_list = []
    for imgs in images_:
        imgs = np.array(Image.open('./img/face_recognition/'+imgs))
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(imgs)[0]
        encode_list.append(encode)
    return encode_list

def recognize_users(cap):
    path = f'{folder_location}img/face_recognition'
    recognized_users = []  # List to store names of recognized users
    images = []
    classNames = []
    myList = os.listdir(path)

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    try:
        with open(f'{folder_location}models/face_rec', 'rb') as file:
            encodeListKnown = pickle.load(file)
    except:
        path = f'{folder_location}img/face_recognition'
        images = []
        classNames = []
        myList = os.listdir(path)
        images = myList

        encodeListKnown = find_encodings(images)
        print(len(encodeListKnown))
        print('Encoding Complete')

        with open(f'{folder_location}models/face_rec', 'wb') as file:
            pickle.dump(encodeListKnown, file)
            file.close()

    _, img = cap.read()
    img = cv2.flip(img, 2)
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndices = np.where(matches)[0]  # Get indices of all matched faces
        
        for matchIndex in matchIndices:
            name = classNames[matchIndex].upper()
            recognized_users.append(name)  # Append recognized user to the list
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (205, 154, 79), 2)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
        
        if not matchIndices.all():
            name = "UNKNOWN"
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (205, 154, 79))
            cv2.putText(img, 'UNKNOWN', (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

    # cv2.imshow("Face Recognition", img)
    # cv2.waitKey(1)

    return recognized_users, img


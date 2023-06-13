from statistics import mode
import numpy as np
import dlib
import cv2


with open('pwd.txt', 'r') as folder_location:
    folder_location = folder_location.read()
eye_contact_list = []
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(f"{folder_location}Face Recognition/shape_predictor_68_face_landmarks.dat")

def eye_contact(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (150, 50, 0), 2)
        landmarks = predictor(gray, face)
        
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray, frame)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray, frame)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

        if gaze_ratio == 0.0:
            return "closed"
        elif gaze_ratio <= 1:
            return "right"
        elif 1 < gaze_ratio < 2.7:
            return "center"
        else:
            return "left"


def get_gaze_ratio(eye_points, facial_landmarks, gray, frame):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)

    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    eye = cv2.resize(gray_eye, None, fx=6, fy=6)
    try:
        gaze_ratio = left_side_white/right_side_white
    except:
        gaze_ratio = 0
    
    return gaze_ratio

def eye_mode(frame):
    global eye_contact_list
    convert = {"right": 1, "center":2, "left": 3}
    data = (eye_contact(frame))
    if len(eye_contact_list) < 25:
        try:
            eye_contact_list.append(convert[data])
        except:
            pass
    else:
        del eye_contact_list[0]
        try:
            eye_contact_list.append(convert[data])
        except:
            pass
        new_data = mode(eye_contact_list)
        for key, value in convert.items():
            if value == new_data:
                return key

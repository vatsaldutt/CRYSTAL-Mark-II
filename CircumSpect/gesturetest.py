import cv2
import mediapipe as mp
import time

# Function to detect hand gestures
def detect_hand_gestures(image, prev_gesture, prev_time):
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        # Convert the image to RGB and process it
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Check if any hand is detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Analyze hand gestures
            # Add your code here to identify hand motions based on the hand landmarks

            # Example: Detect left/right sliding motion
            thumb_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x
            index_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x

            if thumb_x > index_x:
                current_gesture = "Sliding hand to the right"
            else:
                current_gesture = "Sliding hand to the left"

            # Add more conditions to detect other hand gestures

            # Compare with the previous gesture and time
            current_time = time.time()
            if current_gesture != prev_gesture:
                if prev_gesture is not None and (current_time - prev_time) <= 1.5:
                    # Determine the faster motion as the main gesture
                    gesture_speeds = {"Sliding hand to the right": thumb_x - index_x,
                                      "Sliding hand to the left": index_x - thumb_x}
                    main_gesture = max(gesture_speeds, key=gesture_speeds.get)
                    print("Main Gesture:", main_gesture)
                else:
                    # Print the current gesture as the main gesture
                    print("Main Gesture:", current_gesture)

                # Return the current gesture and time as the new previous gesture and time
                return current_gesture, current_time

        # Return None if no hand is detected or not enough time has elapsed
        return prev_gesture, prev_time

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize previous gesture and time as None
prev_gesture = None
prev_time = None

while True:
    # Read frames from the video stream
    ret, frame = cap.read()

    # Stop the loop if no frame is captured
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Detect hand gestures and get the current gesture and time
    current_gesture, current_time = detect_hand_gestures(frame, prev_gesture, prev_time)

    # Display the frame
    cv2.imshow('Hand Gestures', frame)

    # Update the previous gesture and time with the current gesture and time
    prev_gesture = current_gesture
    prev_time = current_time

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

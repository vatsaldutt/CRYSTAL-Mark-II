from CircumSpect.dense import describe_image
from CircumSpect.faces import recognize_users
import cv2
import time

def analyze_image(cap):
    users, recognized = recognize_users(cap)

    captions, annotated = describe_image(recognized)

    if users == []:
        users = "No faces identified"

    if type(users) == list:
        users = ", ".join(users)

    output = f"""Faces: {users}
View description: {", ".join(captions)}"""
    return output, annotated

# cap = cv2.VideoCapture(0)
# while True:
#     output, image = analyze_image(cap)
#     print(output)

#     cv2.imshow("Annotated Image", image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

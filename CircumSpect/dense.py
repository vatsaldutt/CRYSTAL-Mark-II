from matplotlib.patches import Rectangle
from CircumSpect.describe import process_image
import matplotlib.pyplot as plt
from PIL import Image
import ocrmac
import json
import time
import cv2
import os

time.sleep(2)

with open('pwd.txt', 'r') as pwd:
    folder_location = pwd.read()


def crop_and_save_image(img, box, output_path):

    # Convert box coordinates to integers
    box = [int(coord) for coord in box]

    # Crop the image to the specified region of interest
    cropped_img = img[box[1]:box[3], box[0]:box[2]]
    cv2.imwrite(output_path, cropped_img)


def visualize_result(image_file_path, result):
    assert isinstance(result, list)

    og_img = cv2.imread(image_file_path)
    img = cv2.imread(image_file_path)

    captions = []

    for r in result:
        box = r['box']
        caption = r['cap']
        if "<unk>" in caption:
            crop_and_save_image(og_img, box, "ocr.png")
            recognized = ocrmac.OCR('Sample Images/Image.jpeg').recognize()
            caption = caption.replace("<unk>", recognized[0][0])
        cv2.rectangle(img, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])), (0, 0, 255), 2)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(
            box[2]), int(box[1])-50), (200, 200, 200), -1)
        cv2.rectangle(img, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[1])-50), (0, 0, 0), 2)
        cv2.putText(img, caption, (int(box[0]), int(
            box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        captions.append(caption)

    cv2.imwrite("output.png", img)

    return captions, img


def describe_image(frame):
    IMG_FILE_PATH = f'{folder_location}image.png'

    cv2.imwrite(IMG_FILE_PATH, frame)
    process_image(IMG_FILE_PATH, folder_location)

    RESULT_JSON_PATH = f'{folder_location}CircumSpect/result.json'
    with open(RESULT_JSON_PATH, 'r') as f:
        results = json.load(f)

    TO_K = 10
    assert IMG_FILE_PATH in results.keys()
    captions, frame = visualize_result(
        IMG_FILE_PATH, results[IMG_FILE_PATH][:TO_K])

    return captions, frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    time.sleep(2)
    start = time.time()
    while True:
        end = time.time()
        print(end-start)
        _, img = cap.read()
        caption, frame = describe_image(img)
        cv2.imshow("CircumSpect", frame)
        cv2.waitKey(1)
        start = time.time()
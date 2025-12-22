from ocrmac import ocrmac
annotations = ocrmac.OCR('Sample Images/Image.jpeg').recognize()
print(annotations)
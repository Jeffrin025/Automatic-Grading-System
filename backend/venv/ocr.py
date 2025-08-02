from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # use_angle_cls improves rotated text recognition

# Change this to your image filename
image_path = 'image.jpg'

# Perform OCR
results = ocr.ocr(image_path, cls=True)

# Print extracted text
for line in results[0]:
    text = line[1][0]
    print(text)

# import easyocr
# import cv2
# import matplotlib.pyplot as plt

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Optionally, you can apply thresholding or other filters to enhance image quality
#     # _, processed_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
#     return gray_image

# def extract_text_from_image(image_path):
#     reader = easyocr.Reader(['en'], gpu=False)  
#     processed_image = preprocess_image(image_path)
#     plt.imshow(processed_image, cmap='gray')
#     plt.title("Preprocessed Image")
#     plt.axis('off')
#     plt.show()
#     result = reader.readtext(processed_image)
#     return result


# def run_ocr(image_path):
#     extracted_text = extract_text_from_image(image_path)

#     for (bbox, text, prob) in extracted_text:
#         print(f"Detected Text: {text}, Confidence: {prob:.2f}")

# if __name__ == "__main__":
#     image_path = 'img1.jpg'  
#     run_ocr(image_path)

import easyocr
import cv2
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale (optional, but common for OCR preprocessing)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray_image

def extract_text_from_image(image_path):
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=False)

    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Display the preprocessed image
    plt.imshow(processed_image, cmap='gray')
    plt.title("Preprocessed Image")
    plt.axis('off')
    plt.show()

    # Perform OCR on the processed image
    # Note: EasyOCR expects a list of NumPy arrays for input
    result = reader.readtext(processed_image)

    return result

def run_ocr(image_path):
    # Extract text from the image
    extracted_text = extract_text_from_image(image_path)

    # Display the extracted text with confidence scores
    for (bbox, text, prob) in extracted_text:
        print(f"Detected Text: {text}, Confidence: {prob:.2f}")

if __name__ == "__main__":
    # Path to the input image
    image_path = 'images.jpg'  # Update this to your actual image path

    # Run the OCR
    run_ocr(image_path)

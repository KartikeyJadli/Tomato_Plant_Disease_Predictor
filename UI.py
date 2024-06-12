import cv2
from cv2 import VideoCapture
from keras.models import load_model 
import numpy as np
model=load_model("crop_prediction_model.h5",compile=False)
labels=["Tomato Bacterial Spot","Early Blight","Healthy","Late Blight","Leaf Mold","Tomato Septoria leaf spot", "Tomato___Spider_mites Two spotted spider mite","Tomato___Target_Spot","Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]

def classify_image(img):
    img=cv2.resize(img,(256, 256,))
    img = np.reshape(img, (-1, 256, 256, 3))
    type=predict_crop(img)
    return type

def predict_crop(img):
    crop_class=model.predict(img)
    index=np.argmax(crop_class)
    label=labels[index]
    return label

def extract_leaf(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply binary thresholding to get a binary image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Find contours in the binary image 
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create an empty mask
    mask = np.zeros_like(gray)
    # Draw the largest contour on the mask
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    # Extract the leaf using bitwise_and operation
    leaf_extracted = cv2.bitwise_and(image, image, mask=mask)
    return leaf_extracted

def enhance_resolution(image):
    scale_factor=2.0

    interpolation=cv2.INTER_CUBIC
    if image is None:
        raise ValueError("Image not found or path is incorrect")
    
    # Get original dimensions
    height, width = image.shape[:2]
    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    # Resize the image
    enhanced_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation) 
    return enhanced_image
def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from webcam
        try:
            ret, img = cap.read()

            # Display frame
            cv2.imshow('Webcam', img)

            # Wait for a click event (key press)
            key = cv2.waitKey(1) & 0xFF

            # If 'c' key is pressed, capture image
            if key == ord('c'):
                # Save captured image
                img=enhance_resolution(img)
                img=extract_leaf(img)
                print(classify_image(img))
                cv2.imwrite('captured_image.png', img)
            
            # If 'q' key is pressed, exit loop
            elif key == ord('q'):
                break 
        except Exception as e:
            print(e)

    # Release webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
main()

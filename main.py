import cv2
import os

# Load the Haar Cascade model for face detection from the models directory
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')


def detect_faces(image):
    # Convert image to grayscale (Haar Cascade works better on grayscale images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return image


def main():
    # Define the path to the image
    image_path = 'images/your_image.jpg'  # Replace with your actual image name
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return

    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print("Could not load image!")
        return

    # Detect faces in the image
    output_image = detect_faces(image)

    # Show the image with detected faces
    cv2.imshow('Face Detection', output_image)

    # Wait for user interaction and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

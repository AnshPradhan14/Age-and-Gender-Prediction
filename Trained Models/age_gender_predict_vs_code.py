import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

#load trained Keras model
model = load_model('age_gender_model.h5', compile=False)

#prediction function
def predict_age_gender(image_path, model, image_size=128):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size)) / 255.0
    img_exp = np.expand_dims(img, axis=0)

    age_pred, gender_pred = model.predict(img_exp)
    age = int(age_pred[0][0])
    gender = "Female" if np.argmax(gender_pred[0]) == 1 else "Male"

    return age, gender, img

#webcam capture and prediction
def capture_and_predict_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    print("Press 's' to capture, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        cv2.imshow("Webcam - Press 's' to predict", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            img_path = "captured_image.jpg"
            cv2.imwrite(img_path, frame)
            age, gender, img = predict_age_gender(img_path, model)

            plt.imshow(img)
            plt.title(f"Age: {age}, Gender: {gender}")
            plt.axis('off')
            plt.show()

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#file-based prediction using path input
def upload_and_predict_local():
    image_path = input("Enter full path of image: ").strip('"')
    if not os.path.exists(image_path):
        print("File not found.")
        return

    age, gender, img = predict_age_gender(image_path, model)
    plt.imshow(img)
    plt.title(f"Age: {age}, Gender: {gender}")
    plt.axis('off')
    plt.show()

    print(f"Predicted Age: {age}")
    print(f"Predicted Gender: {gender}")

#run options
if __name__ == "__main__":
    print("\nChoose an option:")
    print("1. Predict from webcam")
    print("2. Predict from image file")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        capture_and_predict_webcam()
    elif choice == '2':
        upload_and_predict_local()
    else:
        print("Invalid option.")

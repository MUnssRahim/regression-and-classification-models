import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

catsFolder = os.path.join(os.path.expanduser('~'), 'Desktop', 'cats')
dogsFolder = os.path.join(os.path.expanduser('~'), 'Desktop', 'dogs')

def loadImages(folder, label, images, labels):
    for filename in os.listdir(folder):
        imgPath = os.path.join(folder, filename)
        try:
            img = cv2.imread(imgPath)
            if img is not None:
                imgResized = cv2.resize(img, (128, 128))
                imgFlattened = imgResized.flatten()
                images.append(imgFlattened)
                labels.append(label)
        except Exception as e:
            print(f"Warning: Error: {e}")

images = []
labels = []

loadImages(catsFolder, 0, images, labels)
loadImages(dogsFolder, 1, images, labels)

X = np.array(images)
y = np.array(labels)

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.1, random_state=59)

scaler = StandardScaler()
XTrain = scaler.fit_transform(XTrain)
XTest = scaler.transform(XTest)

model = SVC(kernel='linear')
model.fit(XTrain, yTrain)

yPred = model.predict(XTest)

print("Classification Report:")
print(classification_report(yTest, yPred, target_names=['Cat', 'Dog']))
print("F1 Score (Macro):", f1_score(yTest, yPred, average='macro'))
print("F1 Score (Weighted):", f1_score(yTest, yPred, average='weighted'))

def displayImage(image, label, prediction):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"True Label: {'Cat' if label == 0 else 'Dog'} | Prediction: {'Cat' if prediction == 0 else 'Dog'}")
    plt.axis('off')
    plt.show()

def predictAndDisplayImage(imagePath, model, scaler):
    img = cv2.imread(imagePath)
    if img is None:
        print(f"Error: Unable to read image from {imagePath}")
        return
    imgResized = cv2.resize(img, (128, 128))
    imgFlattened = imgResized.flatten().reshape(1, -1)
    imgScaled = scaler.transform(imgFlattened)
    prediction = model.predict(imgScaled)
    plt.imshow(cv2.cvtColor(imgResized, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {'Cat' if prediction[0] == 0 else 'Dog'}")
    plt.axis('off')
    plt.show()

imagePath = os.path.join(os.path.expanduser('~'), 'Desktop', 'Dog.jpg')
predictAndDisplayImage(imagePath, model, scaler)

import cv2
import io
import os
import tensorflow as tf
from google.cloud import vision_v1p3beta1 as vision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Set the path to your Google Cloud credentials JSON file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'client_key.json'

def load_food_names(food_type):
    """
    Load all known food type names.
    :param food_type: Fruit or Vegetable
    :return: List of food names
    """
    with open(f"{food_type}.dict", 'r') as file:
        names = [line.strip().lower() for line in file]
    return names

def recognize_food_with_google_vision(image_path, food_names, top_n=1, output_image=True, save_to_file=False, resize_output=False):
    img = cv2.imread(image_path)

    # Resize the image if needed
    if resize_output:
        img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1]))

    # Create a Google Vision client
    client = vision.ImageAnnotatorClient()

    # Read the image file
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Perform label detection
    response = client.label_detection(image=image)
    labels = response.label_annotations

    recognized_foods = []

    for label in labels:
        desc = label.description.lower()
        score = round(label.score, 2)

        if desc in food_names:
            recognized_foods.append((desc, score))
            if output_image:
                cv2.putText(img, f"{desc.upper()} ({score})", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 200), 2)

    # Sort recognized foods by confidence score
    recognized_foods.sort(key=lambda x: x[1], reverse=True)

    if save_to_file:
        with open("recognized_foods.txt", "a") as file:
            file.write(f"Image: {image_path}\n")
            for food, score in recognized_foods[:top_n]:
                file.write(f"{food}: {score}\n")
            file.write("\n")

    if output_image:
        cv2.imshow('Recognize & Draw', img)
        cv2.waitKey(0)

    return recognized_foods

def recognize_food_with_tf(image_path, food_names, top_n=1, output_image=True, save_to_file=False, resize_output=False):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, (224, 224))
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)

    # Load a pre-trained TensorFlow model (e.g., MobileNetV2)
    model = tf.keras.applications.MobileNetV2(weights='imagenet')

    # Use the model to predict the class labels
    preds = model.predict(img)
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds.numpy(), top=top_n)[0]

    recognized_foods = []

    for (_, label, score) in decoded_preds:
        desc = label.lower()
        score = round(score, 2)

        if desc in food_names:
            recognized_foods.append((desc, score))

    if save_to_file:
        with open("recognized_foods.txt", "a") as file:
            file.write(f"Image: {image_path}\n")
            for food, score in recognized_foods:
                file.write(f"{food}: {score}\n")
            file.write("\n")

    if output_image:
        img = cv2.imread(image_path)
        if resize_output:
            img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1]))
        for (food, score) in recognized_foods:
            cv2.putText(img, f"{food.upper()} ({score})", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 200), 2)
        cv2.imshow('Recognize & Draw', img)
        cv2.waitKey(0)

    return recognized_foods

def recognize_food_with_pytorch(image_path, food_names, top_n=1, output_image=True, save_to_file=False, resize_output=False):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)

    # Load a pre-trained PyTorch model (e.g., ResNet)
    model = models.resnet50(pretrained=True)
    model = model.eval()

    # Use the model to predict the class labels
    with torch.no_grad():
        output = model(img)
        _, preds = output.topk(top_n)

    recognized_foods = []

    for i in range(top_n):
        label_idx = preds[0][i].item()
        desc = food_names[label_idx]
        score = output[0][label_idx].item()

        recognized_foods.append((desc, score))

    if save_to_file:
        with open("recognized_foods.txt", "a") as file:
            file.write(f"Image: {image_path}\n")
            for food, score in recognized_foods:
                file.write(f"{food}: {score}\n")
            file.write("\n")

    if output_image:
        img = cv2.imread(image_path)
        if resize_output:
            img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1]))
        for (food, score) in recognized_foods:
            cv2.putText(img, f"{food.upper()} ({score:.2f})", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 200), 2)
        cv2.imshow('Recognize & Draw', img)
        cv2.waitKey(0)

    return recognized_foods

if __name__ == "__main__":
    FOOD_TYPE = 'Fruit'  # 'Vegetable'
    list_foods = load_food_names(FOOD_TYPE)
    print(list_foods)

    # Single image processing
    image_path = '1.jpg'  # Update with your image path

    # Recognize with Google Vision
    recognized_foods_google = recognize_food_with_google_vision(image_path, list_foods, top_n=3, output_image=True, save_to_file=True, resize_output=True)
    print("Recognized Foods (Google Vision):", recognized_foods_google)

    # Recognize with TensorFlow
    recognized_foods_tf = recognize_food_with_tf(image_path, list_foods, top_n=3, output_image=True, save_to_file=True, resize_output=True)
    print("Recognized Foods (TensorFlow):", recognized_foods_tf)

    # Recognize with PyTorch
    recognized_foods_pytorch = recognize_food_with_pytorch(image_path, list_foods, top_n=3, output_image=True, save_to_file=True, resize_output=True)
    print("Recognized Foods (PyTorch):", recognized_foods_pytorch)

    # Batch processing multiple images
    image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']  # Replace with a list of image paths
    for image_path in image_paths:
        print(f"Image: {image_path}")
        # Recognize with Google Vision
        recognized_foods_google = recognize_food_with_google_vision(image_path, list_foods, top_n=3, output_image=False, save_to_file=True, resize_output=False)
        print("Recognized Foods (Google Vision):", recognized_foods_google)
        # Recognize with TensorFlow
        recognized_foods_tf = recognize_food_with_tf(image_path, list_foods, top_n=3, output_image=False, save_to_file=True, resize_output=False)
        print("Recognized Foods (TensorFlow):", recognized_foods_tf)
        # Recognize with PyTorch
        recognized_foods_pytorch = recognize_food_with_pytorch(image_path, list_foods, top_n=3, output_image=False, save_to_file=True, resize_output=False)
        print("Recognized Foods (PyTorch):", recognized_foods_pytorch)

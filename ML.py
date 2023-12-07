import os
import urllib.request
import tarfile
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Step 1: Download and Extract Dataset
url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
download_path = "VOCtrainval_11-May-2012.tar"

urllib.request.urlretrieve(url, download_path)

with tarfile.open(download_path, "r") as tar:
    tar.extractall()

os.remove(download_path)

# Step 2: Explore the Dataset (optional)

# Step 3: Preprocess the Dataset
def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall(".//object"):
        obj_info = {}
        obj_info["name"] = obj.find("name").text
        bbox = obj.find("bndbox")
        obj_info["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_info)

    return objects

def preprocess_data(image_dir, annotation_dir):
    images = []
    labels = []

    for annotation_file in os.listdir(annotation_dir):
        image_id = os.path.splitext(annotation_file)[0]
        image_path = os.path.join(image_dir, image_id + ".jpg")
        annotation_path = os.path.join(annotation_dir, annotation_file)

        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)

        annotations = parse_annotation(annotation_path)

        for obj in annotations:
            label = obj["name"]
            bbox = obj["bbox"]
            images.append(img_array)
            labels.append({"label": label, "bbox": bbox})

    return np.array(images), labels

image_dir = "VOCdevkit/VOC2012/JPEGImages"
annotation_dir = "VOCdevkit/VOC2012/Annotations"

images, labels = preprocess_data(image_dir, annotation_dir)

# Step 4: Build the Model
def build_model():
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(20, activation="softmax")
    ])

    return model

model = build_model()

# Step 5: Compile and Train the Model
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

def generator(images, labels, batch_size=32):
    while True:
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            yield np.array(batch_images), {"output": batch_labels}

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(generator(X_train, y_train), epochs=5, steps_per_epoch=len(X_train) // 32,
                    validation_data=generator(X_val, y_val), validation_steps=len(X_val) // 32)

#Step 6 - Evaluation
def calculate_iou(box1, box2):
    # Calculate the intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate intersection area
    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    # Calculate union area
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area

    return iou

def evaluate_model(model, X_val, y_val):
    total_iou = 0.0
    total_samples = len(X_val)

    for i in range(total_samples):
        image = np.expand_dims(X_val[i], axis=0)
        true_box = y_val[i]["bbox"]

        # Predict bounding box
        predicted_box = model.predict(image)[0]

        # Convert from normalized coordinates to original image coordinates
        predicted_box = [
            int(predicted_box[0] * 224),
            int(predicted_box[1] * 224),
            int(predicted_box[2] * 224),
            int(predicted_box[3] * 224),
        ]

        # Calculate IoU for this sample
        iou = calculate_iou(true_box, predicted_box)
        total_iou += iou

    # Calculate average IoU
    average_iou = total_iou / total_samples

    return average_iou

# Evaluate the trained model on the validation set
iou_score = evaluate_model(model, X_val, y_val)
print(f"Average IoU on the validation set: {iou_score}")
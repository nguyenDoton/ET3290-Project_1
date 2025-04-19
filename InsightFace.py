from google.colab import drive
import cv2
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
from insightface.app import FaceAnalysis
import torchvision.transforms.functional as TF
import random
from torchvision import transforms
from PIL import Image
from PIL import ImageFilter

import os
drive.mount('/content/drive')
dataset_path = "/content/drive/MyDrive/Training Dataset"
print("People found:", os.listdir(dataset_path))

def augment_image_for_test(image, rotation_deg=10, noise_std=0.05, blur_radius=1.5, brightness_factor=1.0):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_pil = Image.fromarray(image_rgb)

    image_pil = TF.rotate(image_pil, rotation_deg)

    image_pil = TF.adjust_brightness(image_pil, brightness_factor)

    if blur_radius > 0:
        image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    tensor_img = TF.to_tensor(image_pil)
    noise = torch.randn_like(tensor_img) * noise_std
    noisy_tensor = torch.clamp(tensor_img + noise, 0.0, 1.0)

    noisy_rgb = (noisy_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    noisy_bgr = cv2.cvtColor(noisy_rgb, cv2.COLOR_RGB2BGR)

    return noisy_bgr




app = FaceAnalysis(name= 'buffalo_l') 
app.prepare(ctx_id=0 if torch.cuda.is_available() else -1 ,det_thresh=0.25 )

embedding_db = defaultdict(list)

def build_embedding_db_insightface(dataset_path, augment_times=1):
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                try:
                    img_cv2 = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)

                    for _ in range(augment_times):
                        aug_pil = augment_image(img_pil)
                        aug_np = np.array(aug_pil)  # Convert back to NumPy
                        aug_bgr = cv2.cvtColor(aug_np, cv2.COLOR_RGB2BGR)

                        faces = app.get(aug_bgr)
                        if faces:
                            emb = faces[0].embedding
                            embedding_db[person].append(emb)

                except Exception as e:
                    print(f"Skipping {img_path}: {e}")

def recognize_face_batch_insightface(folder_path, threshold=0.7):
    true_labels = []
    predicted_labels = []

    for person in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person)
        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    augment_img = augment_image_for_test(img, rotation_deg=30, noise_std=0.07, blur_radius= 1.5, brightness_factor = 1.0)
                    faces = app.get(augment_img)
                    # faces = app.get(img)
                    if not faces:
                        print(f"[{img_path}] → No face detected")
                        continue

                    query_emb = faces[0].embedding.reshape(1, -1)
                    best_match = None
                    best_score = -1

                    for db_person, db_embeddings in embedding_db.items():
                        for ref_emb in db_embeddings:
                            sim = cosine_similarity(query_emb, ref_emb.reshape(1, -1))[0][0]
                            if sim > best_score:
                                best_score = sim
                                best_match = db_person

                    true_labels.append(person)

                    if best_score < threshold:
                        predicted_labels.append("Unknown")
                        print(f"[{img_path}] → Unknown face (sim: {best_score:.2f}) | Actual: {person}")
                    else:
                        predicted_labels.append(best_match)
                        print(f"[{img_path}] → Predicted: {best_match} | Actual: {person} | Similarity: {best_score:.2f}")

                except Exception as e:
                    print(f"Skipping {img_path}: {e}")

    # Evaluation
    filtered_true = [t for t, p in zip(true_labels, predicted_labels) if p != "Unknown"]
    filtered_pred = [p for p in predicted_labels if p != "Unknown"]
    if filtered_true:
        acc = accuracy_score(filtered_true, filtered_pred)
        f1 = f1_score(filtered_true, filtered_pred, average='weighted')
        print(f"\nFiltered Accuracy (excluding Unknowns): {acc:.4f}")
        print(f"F1 Score (excluding Unknowns): {f1:.4f}")
    else:
        print("No confident predictions made.")
build_embedding_db_insightface("/content/drive/MyDrive/Training Dataset")
recognize_face_batch_insightface("/content/drive/MyDrive/test")

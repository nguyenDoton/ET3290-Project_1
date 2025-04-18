from deepface import DeepFace
import os
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
from google.colab import drive
drive.mount('/content/drive', force_remount = True)
dataset_path = "/content/drive/MyDrive/Training Dataset"
print("People found:", os.listdir(dataset_path))
import cv2
import torch
from PIL import Image
from PIL import ImageFilter
from torchvision.transforms import functional as TF
import torchvision.transforms as transforms

embedding_db = defaultdict(list)

def augment_image_for_test(image, rotation_deg=10, noise_std=0.05,blur_radius=1.5, brightness_factor=1.0):
    
    image = TF.rotate(image, rotation_deg)

    image = TF.adjust_brightness(image, brightness_factor)

    if blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    tensor_img = TF.to_tensor(image)
    noise = torch.randn_like(tensor_img) * noise_std
    noisy_img = tensor_img + noise
    noisy_img = torch.clamp(noisy_img, 0.0, 1.0)


    return TF.to_pil_image(noisy_img)





def build_embedding_db_deepface(dataset_path):
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                try:
                    embedding = DeepFace.represent(
                        img_path=img_path,
                        model_name='VGG-Face',
                        enforce_detection=False
                    )[0]["embedding"]
                    embedding_db[person].append(np.array(embedding))
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")

def recognize_face_batch_deepface(folder_path, threshold=0.7):
    true_labels = []
    predicted_labels = []

    for person in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person)
        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                try:
                    # Read and convert to RGB
                    img_bgr = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(img_rgb)
                    # aug_img = augment_image_for_test(pil_image, rotation_deg=0, noise_std=0)
                    rotation_deg =30
                    noise_std = 0.07
                    blur = 1.5
                    brightness = 1.0
                    # img_aug_np = np.array(aug_img)
                    if rotation_deg == 0 and noise_std == 0:
                            img_aug_np = np.array(pil_image)  # No augmentation, use original
                    else:
                           aug_img = augment_image_for_test(pil_image, rotation_deg, noise_std,blur,brightness)
                           img_aug_np = np.array(aug_img)

                    # Run DeepFace on augmented image
                    result = DeepFace.represent(
                        img_aug_np,
                        model_name='VGG-Face',
                        enforce_detection=False
                    )

                    if not result:
                        print(f"[{img_path}] → No face detected")
                        continue

                    query_emb = np.array(result[0]["embedding"]).reshape(1, -1)
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
build_embedding_db_deepface("/content/drive/MyDrive/Training Dataset")
recognize_face_batch_deepface("/content/drive/MyDrive/test")
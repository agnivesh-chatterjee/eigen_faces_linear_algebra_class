import os
import cv2
import numpy as np
from collections import defaultdict

def load_att_faces_flat(root_dir, train_per_person=6):
    images_by_person = defaultdict(list)

    print("Scanning folder:", root_dir)

    for fname in os.listdir(root_dir):
        fname_lower = fname.lower()

        if not fname_lower.endswith((".pgm", ".png", ".jpg", ".jpeg")):
            continue

        # Remove extension safely
        name = os.path.splitext(fname)[0]

        # Expect format: x_y  → y is person id
        if "_" not in name:
            print("Skipping (no underscore):", fname)
            continue

        try:
            person_id = int(name.split("_")[-1])
        except ValueError:
            print("Skipping (bad label):", fname)
            continue

        img_path = os.path.join(root_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Skipping (cv2 failed):", fname)
            continue

        img = img.astype(np.float64).flatten()
        images_by_person[person_id].append(img)

    print("People detected:", sorted(images_by_person.keys()))

    X_train, y_train, X_test, y_test = [], [], [], []

    for label, person_id in enumerate(sorted(images_by_person.keys())):
        imgs = images_by_person[person_id]

        print(f"Person {person_id}: {len(imgs)} images")

        if len(imgs) < train_per_person:
            print(f"⚠️ Not enough images for person {person_id}")
            continue

        for i, img in enumerate(imgs):
            if i < train_per_person:
                X_train.append(img)
                y_train.append(label)
            else:
                X_test.append(img)
                y_test.append(label)

    X_train = np.array(X_train).T
    X_test = np.array(X_test).T

    print("Final shapes:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    return X_train, np.array(y_train), X_test, np.array(y_test)

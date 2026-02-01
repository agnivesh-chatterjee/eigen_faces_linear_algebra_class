from preprocessing import load_att_faces_flat
from pca import Eigenfaces
from recognition import EigenfaceRecognizer
import numpy as np

DATASET_PATH = "archive(4)"

X_train, y_train, X_test, y_test = load_att_faces_flat(DATASET_PATH)

# 🔍 SANITY CHECKS — PUT THEM HERE
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Unique train labels:", len(set(y_train)))
print("Unique test labels:", len(set(y_test)))

# ---- rest of your code ----

K = 40
model = Eigenfaces(num_components=K)
model.fit(X_train)

recognizer = EigenfaceRecognizer(model)
recognizer.fit(X_train, y_train)

preds = recognizer.predict(X_test)
accuracy = np.mean(preds == y_test)
print(f"Accuracy with K={K}: {accuracy*100:.2f}%")


import face_make # if it's in another file
# OR just call it directly if it's in the same file

face_make.show_eigenfaces(model, h=70, w=80, num=10)

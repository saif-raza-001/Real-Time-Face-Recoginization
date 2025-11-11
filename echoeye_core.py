import cv2
import numpy as np
import os
import json

# -------------------------
# Constants
# -------------------------
DEFAULT_DATASET_PATH = "./face_dataset/"
DEFAULT_ROLES_FILE = "roles.json"
CASCADE_PATH = "haarcascade_frontalface_alt.xml"


# -------------------------
# Utility Functions
# -------------------------
def load_roles(path=DEFAULT_ROLES_FILE):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_roles(roles, path=DEFAULT_ROLES_FILE):
    with open(path, "w") as f:
        json.dump(roles, f, indent=4)


# -------------------------
# Face Data Collection
# -------------------------
def collect_face_data(name, save_path=DEFAULT_DATASET_PATH, max_images=50, cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    skip = 0
    face_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            continue

        # Sort and select the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]

        try:
            face_section = cv2.resize(face_section, (100, 100))
        except Exception as e:
            continue

        if skip % 10 == 0:
            face_data.append(face_section)
            print(f"[INFO] Collected image {len(face_data)}/{max_images}")

        skip += 1

        # Show frames
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Collecting Face Data", frame)
        cv2.imshow("Face", face_section)

        # Stop if 'q' or enough images collected
        if cv2.waitKey(1) & 0xFF == ord('q') or len(face_data) >= max_images:
            break

    # Save data
    face_data = np.array(face_data).reshape((len(face_data), -1))
    np.save(os.path.join(save_path, name.lower()), face_data)

    print(f"[SUCCESS] Saved dataset: {name.lower()}.npy")

    cap.release()
    cv2.destroyAllWindows()


# -------------------------
# Real-Time Face Recognition
# -------------------------
def recognize_faces(dataset_path=DEFAULT_DATASET_PATH, roles_path=DEFAULT_ROLES_FILE, cam_index=0):
    def distance(v1, v2):
        return np.linalg.norm(v1 - v2)

    def knn(train, test, k=5):
        dist = []
        for i in range(train.shape[0]):
            ix = train[i, :-1]
            iy = train[i, -1]
            d = distance(test, ix)
            dist.append([d, iy])
        dk = sorted(dist, key=lambda x: x[0])[:k]
        labels = np.array(dk)[:, -1]
        output = np.unique(labels, return_counts=True)
        return output[0][np.argmax(output[1])]

    # Load dataset
    face_data = []
    labels = []
    names = {}
    class_id = 0

    for file in os.listdir(dataset_path):
        if file.endswith(".npy"):
            name = file[:-4].lower()
            data = np.load(os.path.join(dataset_path, file))
            face_data.append(data)
            labels.append(class_id * np.ones((data.shape[0],)))
            names[class_id] = name
            class_id += 1

    if not face_data:
        print("[ERROR] No face data found.")
        return

    # Prepare training data
    face_dataset = np.concatenate(face_data, axis=0)
    face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
    trainset = np.concatenate((face_dataset, face_labels), axis=1)

    roles = load_roles(roles_path)

    # Start webcam
    cap = cv2.VideoCapture(cam_index)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    print("[INFO] Starting recognition...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            offset = 10
            face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]

            try:
                face_section = cv2.resize(face_section, (100, 100))
            except:
                continue

            out = knn(trainset, face_section.flatten())
            predicted_name = names.get(int(out), "Unknown").lower()

            role = roles.get(predicted_name, "Intruder")

            if role.lower() == "intruder" or predicted_name == "unknown":
                display_name = "Intruder"
                color = (0, 0, 255)
            else:
                display_name = f"{predicted_name.capitalize()} ({role})"
                color = (0, 255, 0) if role == "Admin" else (255, 255, 0)

            # Display
            cv2.putText(frame, display_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

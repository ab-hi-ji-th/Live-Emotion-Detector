import cv2
import torch 
from transformers import ViTForImageClassification, ViTFeatureExtractor

model_path = "C:\\Users\\Abhijith lappy\\PycharmProjects\\Emotion Detector Live\\vit_emotion_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=7)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

def preprocess_frame(face_region):
    face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
    resized_face = cv2.resize(face_rgb, (224, 224))
    inputs = feature_extractor(images=resized_face, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()} 
    return inputs

def predict_emotion(face_region):
    inputs = preprocess_frame(face_region)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    emotion = emotion_labels[predicted_class]
    return emotion

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not found.")
    exit()

print("Press 'P' to start/stop emotion detection.")
print("Press 'Q' to quit the live emotion recognizer.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

detect_emotion = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if detect_emotion:
            face_region = frame[y:y + h, x:x + w]
            emotion = predict_emotion(face_region)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame, "Press 'P' to toggle detection. Press 'Q' to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Live Emotion Recognizer", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        detect_emotion = not detect_emotion

cap.release()
cv2.destroyAllWindows()

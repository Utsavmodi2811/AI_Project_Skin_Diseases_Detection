import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

def focal_loss(y_true, y_pred, gamma=2., alpha=0.25):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_true = tf.cast(y_true, tf.float32)
    cross_entropy = -y_true * tf.math.log(y_pred)
    loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
    return tf.reduce_mean(loss)

# Use the exact class label order and format as specified by the user
class_labels = [
    "Eczema",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma (BCC)",
    "Melanocytic Nevi (NV)",
    "Benign Keratosis-like Lesions (BKL)",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Seborrheic Keratoses and other Benign Tumors",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Warts Molluscum and other Viral Infections"
]

model_path = "C:/D/My learnings/AI_Project_Skin_diseases_detection/Code/skin_disease_model.h5"

try:
    model = load_model(model_path, custom_objects={'focal_loss': focal_loss}, compile=False)
    print("Model loaded successfully.")
except ValueError as e:
    print("Error loading model:", e)
    print("You may need to check the version of TensorFlow/Keras or re-save the model with the correct loss.")

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    confidence = np.max(prediction)
    predicted_index = int(np.argmax(prediction))
    predicted_label = class_labels[predicted_index]

    print(f"\n🧪 Predicted Class: {predicted_index+1}. {predicted_label}")
    print(f"📈 Confidence: {confidence * 100:.2f}%\n")

    print("📊 Class Probabilities:")
    if len(prediction[0]) != len(class_labels):
        print(f"Model output ({len(prediction[0])}) does not match number of class labels ({len(class_labels)}). Please check your model and class_labels list.")
        print("Model output shape:", prediction[0].shape)
        print("Class labels count:", len(class_labels))
    else:
        for i, prob in enumerate(prediction[0]):
            print(f"{i+1}. {class_labels[i]}: {prob * 100:.2f}%")

if __name__ == "__main__":
    img_path = "C:/D/My learnings/Project/AI/ISIC_0025767.jpg"  # Update this path as needed
    predict_image(img_path)

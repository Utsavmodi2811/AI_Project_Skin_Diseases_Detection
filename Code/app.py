import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import pandas as pd
import plotly.graph_objects as go

def focal_loss(y_true, y_pred, gamma=2., alpha=0.25):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_true = tf.cast(y_true, tf.float32)
    cross_entropy = -y_true * tf.math.log(y_pred)
    loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
    return tf.reduce_mean(loss)

model = load_model(
    "C:/D/My learnings/AI_Project_Skin_diseases_detection/Code/skin_disease_model.h5",
    custom_objects={'focal_loss': focal_loss},
    compile=False
)

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

# Add advanced custom CSS for fonts and layout
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Roboto:wght@400;500&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"]  {
            font-family: 'Montserrat', 'Roboto', sans-serif !important;
        }
        .main-title {
            font-size: 40px;
            font-weight: 800;
            color: #2b6cb0;
            margin-bottom: 0.2em;
            letter-spacing: 1px;
            text-align: center;
            background: linear-gradient(90deg, #e0eafc 0%, #cfdef3 100%);
            border-radius: 16px;
            padding: 0.5em 0;
            box-shadow: 0 2px 8px rgba(44, 62, 80, 0.08);
        }
        .header-logo {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: #2b6cb0;
            display: inline-block;
            margin-right: 16px;
            vertical-align: middle;
        }
        .sidebar-content {
            background: #f7fafc;
            border-radius: 16px;
            padding: 1.5em 1em 1em 1em;
            margin-bottom: 1em;
            box-shadow: 0 2px 8px rgba(44, 62, 80, 0.06);
        }
        .sidebar-title {
            font-size: 26px;
            font-weight: 700;
            color: #2b6cb0;
            margin-bottom: 0.2em;
        }
        .sidebar-desc {
            font-size: 16px;
            color: #444;
        }
        .pred-card {
            background: linear-gradient(90deg, #f8ffae 0%, #43c6ac 100%);
            border-radius: 16px;
            padding: 1.2em 1em;
            margin-bottom: 1em;
            box-shadow: 0 2px 12px rgba(44, 62, 80, 0.10);
            font-size: 1.3em;
            font-weight: 700;
            color: #155724;
            text-align: center;
        }
        .confidence-card {
            background: #e0eafc;
            border-radius: 16px;
            padding: 1.2em 1em;
            margin-bottom: 1em;
            box-shadow: 0 2px 12px rgba(44, 62, 80, 0.10);
            font-size: 1.2em;
            font-weight: 600;
            color: #2b6cb0;
            text-align: center;
        }
        .prob-card {
            background: #fff;
            border-radius: 16px;
            padding: 1.5em 1em 1em 1em;
            margin-bottom: 1em;
            box-shadow: 0 2px 12px rgba(44, 62, 80, 0.10);
        }
        .prob-list {
            font-size: 18px;
            margin-left: 1em;
            font-family: 'Roboto', sans-serif;
        }
        .prob-list-row {
            padding: 0.3em 0.5em;
            border-radius: 8px;
            transition: background 0.2s;
        }
        .prob-list-row:hover {
            background: #e0eafc;
        }
        .section-title {
            font-size: 22px;
            font-weight: 700;
            color: #1a202c;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }
        .footer {
            background: #f7fafc;
            border-radius: 12px;
            padding: 0.7em 1em;
            font-size: 15px;
            color: #888;
            text-align: center;
            margin-top: 2em;
        }
    </style>
""", unsafe_allow_html=True)

# Custom header with logo placeholder
st.markdown("""
<div style='display: flex; align-items: center; justify-content: center; margin-bottom: 1em;'>
    <div class='header-logo'></div>
    <span class='main-title'>Skin Disease Detection Results</span>
</div>
""", unsafe_allow_html=True)

# Sidebar styling
st.sidebar.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='sidebar-title'>🧬 Skin Disease Classifier</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='sidebar-desc'>Upload a skin image to get the predicted disease and its accuracy.</p>", unsafe_allow_html=True)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    img = img.resize((299, 299)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    confidence = float(np.max(prediction))
    predicted_index = int(np.argmax(prediction))
    predicted_label = class_labels[predicted_index]

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"<div class='pred-card'>🩺 <b>Predicted Class:</b><br>{predicted_index+1}. {predicted_label}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='confidence-card'>Confidence<br><b>{confidence * 100:.2f}%</b></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<span class='section-title'>🔬 Class Probabilities:</span>", unsafe_allow_html=True)

    if len(prediction[0]) != len(class_labels):
        st.error(f"Model output ({len(prediction[0])}) does not match number of class labels ({len(class_labels)}). Please check your model and class_labels list.")
        st.write("Model output shape:", prediction[0].shape)
        st.write("Class labels count:", len(class_labels))
    else:
        prob_df = pd.DataFrame({
            'Class': [f"{i+1}. {label}" for i, label in enumerate(class_labels)],
            'Probability (%)': [round(prob * 100, 2) for prob in prediction[0]]
        })
        prob_df = prob_df.sort_values('Probability (%)', ascending=True)  # For horizontal bar chart
        st.markdown("<div class='prob-card'>", unsafe_allow_html=True)
        # Plotly horizontal bar chart
        fig = go.Figure(go.Bar(
            x=prob_df['Probability (%)'],
            y=prob_df['Class'],
            orientation='h',
            marker=dict(color='#2b6cb0'),
            hovertemplate='%{y}: %{x:.2f}%<extra></extra>'
        ))
        fig.update_layout(
            xaxis_title='Probability (%)',
            yaxis_title='',
            plot_bgcolor='#fff',
            paper_bgcolor='#fff',
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(family='Montserrat, Roboto, sans-serif', size=16),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        # Improved probability list with better hover
        st.markdown(
            "<div class='prob-list'>" +
            "".join([
                f"<div class='prob-list-row' style='color:#222;background:rgba(224,234,252,0);' onmouseover=\"this.style.background='#e0eafc';this.style.color='#1a202c'\" onmouseout=\"this.style.background='rgba(224,234,252,0)';this.style.color='#222'\">{row['Class']}: <b>{row['Probability (%)']:.2f}%</b></div>"
                for _, row in prob_df.sort_values('Probability (%)', ascending=False).iterrows()
            ]) +
            "</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='footer'>Model: Xception | Input size: 299x299 | Custom Focal Loss | Trained on 10 skin disease classes</div>", unsafe_allow_html=True)

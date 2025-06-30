# 🧬 AI Project: Skin Diseases Detection

A modern deep learning web app for automatic skin disease classification using the Xception model and Streamlit. Upload a skin image and get instant, visually rich predictions for 10 common skin conditions.

---

## 🚀 Features
- **Deep Learning Model:** Xception architecture, trained on 10 skin disease classes
- **Beautiful UI:** Modern, responsive Streamlit interface with custom styling and interactive charts
- **Instant Prediction:** Upload an image and get class probabilities and confidence
- **Easy to Run:** One-command launch, no web dev required

---

## 🖼️ Demo
> ![image](https://github.com/user-attachments/assets/21d9e079-cb2a-4f6f-87ed-a1bf3b7d90cc)

---

## 🛠️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Utsavmodi2811/AI_Project_Skin_Diseases_Detection.git
   cd AI_Project_Skin_Diseases_Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   _If you don't have a requirements.txt, install manually:_
   ```bash
   pip install streamlit tensorflow pillow numpy pandas plotly
   ```

3. **Download the model file:**
   - Download `skin_disease_model.h5` from [this link](MODEL_DOWNLOAD_LINK_HERE) and place it in `Code/`.
   - Download `class_labels.pkl` from [this link](CLASS_LABELS_LINK_HERE) and place it in `Code/` (if not already present).

---

## 💻 How to Run

```bash
streamlit run Code/app.py
```
- Open the provided local URL in your browser.
- Upload a skin image and view predictions instantly!

---

## 🏷️ Example Usage

```python
# For command-line prediction (see Code/prediction.py):
python Code/prediction.py
```

---

## 🧑‍💻 Technologies Used
- Python 3
- Streamlit
- TensorFlow / Keras
- Xception Model
- Plotly (for interactive charts)
- Pillow, NumPy, Pandas

---

## 📂 Project Structure
```
AI_Project_Skin_Diseases_Detection/
├── Code/
│   ├── app.py              # Streamlit web app
│   ├── prediction.py       # Command-line prediction script
│   ├── skin_disease_model.h5  # (Download separately)
│   └── class_labels.pkl    # Class label mapping
├── Example_images/         # Example images for testing
├── Notebook/               # Model training notebooks
├── Documentation/          # Reports and presentations
└── README.md
```

---

## 🙏 Credits
- Developed by [Utsav Modi](https://github.com/Utsavmodi2811)
- Model architecture: Xception
- UI: Streamlit + Plotly

---

## 📢 Notes
- The model file is **not included** in this repo due to GitHub's file size limits. Please download it from the provided link.
- For any issues, open an issue or pull request on GitHub.

---

_Star ⭐ this repo if you found it useful!_

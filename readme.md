# 📸 CyFuture Face Recognition Demo

Welcome to the **CyFuture** Face Recognition Demo! This repository contains:

- A **dataset** of Yale faces used to train embeddings
- A Jupyter notebook (`cyfuture.ipynb`) showing preprocessing, embedding extraction, and KNN training
- A Streamlit app (`app.py`) for interactive face recognition via upload or webcam
- A `req.txt` listing Python dependencies

---

## 📁 Repository Structure

```
CyFuture/
├── Dataset/                    # Yale face images (PNG)
│   ├── yaleB01_P00A+000E+00.png
│   ├── yaleB01_P00A+005E+00.png
│   └── ... (total ~38 subjects × multiple poses)
│
├── env/                        # (ignored) your Python venv
├── Model.zip                   # (ignored) pretrained weights (use Git LFS or download separately)
│
├── app.py                      # Streamlit face‐recognition demo
├── cyfuture.ipynb              # Notebook: data prep, FaceNet embed, KNN train
├── req.txt                     # pip install requirements
└── .gitignore
```

---

## 🗂 Dataset

The `Dataset/` folder contains the Yale Face Database images:

- **Subjects**: 38 identities labelled `yaleB01` through `yaleB38`.
- **Poses**: Multiple images per subject, different head rotations (`± angles`) and lighting conditions.
- **Format**: Grayscale PNG, resolution ~100×100.

These images are used to compute 512‑dim FaceNet embeddings and train a cosine‑distance KNN classifier.

---

## 🛠 Installation

1. **Clone this repo**:
   ```bash
   git clone https://github.com/Yuvraj171/Cyfuture_21103009.git
   cd Cyfuture_21103009
   ```

2. **Create a Python environment** (recommended):
   ```bash
   python3 -m venv env
   source env/bin/activate     # macOS/Linux
   .\\env\\Scripts\\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r req.txt
   ```

4. **(Optional) Download pretrained weights**

   The Streamlit app expects a FaceNet checkpoint `facenet_yale_weights.pt` under `Model/`. You can either
   - Use Git LFS to fetch large files, or
   - Download and unzip `Model.zip` manually into `Model/`.

---

## 📓 Running the Jupyter Notebook

1. Activate your environment:
   ```bash
   source env/bin/activate
   ```
2. Launch Jupyter:
   ```bash
   jupyter notebook cyfuture.ipynb
   ```
3. **Notebook Overview**:
   - **Data loading**: reads `Dataset/` images
   - **FaceNet backbone**: loads pretrained InceptionResnetV1
   - **Embedding extraction**: crops & aligns faces via MTCNN + computes 512-d embeddings
   - **KNN training**: fits `sklearn.neighbors.KNeighborsClassifier` with cosine distance
   - **Evaluation**: shows accuracy and nearest neighbor distances

Use the notebook cells to retrain on your own gallery or adjust thresholds.

---

## 🚀 Running the Streamlit App

Streamlit provides an interactive UI for face recognition.

1. Activate your environment:
   ```bash
   source env/bin/activate
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. In your browser, you can:
   - **Upload Image**: Drag‑and‑drop a PNG/JPG photo
   - **Use Webcam**: Take a live picture

The app will:
- Detect all faces using MTCNN
- Crop & align each face
- Compute embeddings via FaceNet
- Perform 1‑NN lookup with cosine distance
- Annotate the image with bounding boxes & predicted subject (or "Unknown")
- Display a face‑by‑face summary table

**Fixed settings** (`app.py`)—no sliders:
- `UNKNOWN_THRESHOLD = 0.8`
- `TOP_K = 1`

---

## 📋 Requirements

The `req.txt` includes:
```
streamlit
facenet-pytorch
torch
torchvision
numpy
Pillow
scikit-learn
joblib
```

Install with:
```bash
pip install -r req.txt
```

---

## 🔧 Troubleshooting

- If no faces are detected, try a clearer image, adjust lighting, or lower MTCNN thresholds in `load_models()`.
- If the KNN model unpickling errors arise, ensure scikit-learn versions match (trained on `1.2.2` vs installed `1.6.1`).
- Use Git LFS for large model files (>100 MB).

---

## 🎯 Next Steps

- **Enroll new subjects** on‑the‑fly by adding embeddings and retraining (or app‑side append).
- **Integrate FAISS** for very large galleries.
- **Quantize** FaceNet for faster CPU inference.
- **Monitor** inference times & accuracy with Weights & Biases or MLflow.

---

Happy face‐recognizing! 👋


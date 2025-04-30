# app.py
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pickle, joblib, torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# ─── 1) LOAD KNN, SUBJECT NAMES & LABELS ────────────────────────
@st.cache_resource(show_spinner=False)
def load_knn_and_data():
    data   = pickle.load(open("Model/augmented_embeddings.pkl","rb"))
    labels = data["labels"]               # array of class‐IDs (0…37) per embedding
    knn    = joblib.load("Model/knn_augmented_cosine.joblib")
    # your 38 Yale IDs in the exact order you trained on:
    subjects = [
        "yaleB01","yaleB02","yaleB03","yaleB04","yaleB05","yaleB06","yaleB07",
        "yaleB08","yaleB09","yaleB10","yaleB11","yaleB12","yaleB13","yaleB14",
        "yaleB15","yaleB16","yaleB17","yaleB18","yaleB19","yaleB20","yaleB21",
        "yaleB22","yaleB23","yaleB24","yaleB25","yaleB26","yaleB27","yaleB28",
        "yaleB29","yaleB30","yaleB31","yaleB32","yaleB33","yaleB34","yaleB35",
        "yaleB36","yaleB37","yaleB38"
    ]
    return knn, labels, subjects

knn, labels, subjects = load_knn_and_data()

# ─── 2) LOAD DETECTOR, ALIGNER & FACEBACKBONE ───────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FaceNet backbone
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # detector: finds all faces (lenient thresholds)
    mtcnn_detect = MTCNN(
        image_size=160, margin=0,
        thresholds=[0.3, 0.3, 0.3],
        min_face_size=20, factor=0.709,
        keep_all=True, device=device
    )
    # aligner: single‐face crops → always one 3×160×160 tensor
    mtcnn_align = MTCNN(
        image_size=160, margin=0,
        keep_all=False, device=device
    )

    return mtcnn_detect, mtcnn_align, resnet, device

mtcnn_detect, mtcnn_align, resnet, device = load_models()

# ─── 3) FIXED SETTINGS ───────────────────────────────────────────
UNKNOWN_THRESHOLD = 0.8   # above this cosine‐dist → Unknown
TOP_K             = 1     # 1‐NN

# ─── 4) STREAMLIT UI ─────────────────────────────────────────────
st.title("📸 Face Recognition Demo")

# upload vs webcam
mode = st.radio("Input source", ["Upload Image","Use Webcam"])
if mode == "Upload Image":
    up = st.file_uploader("Upload a face image", type=["png","jpg","jpeg"])
    if not up:
        st.info("Please upload an image to begin.")
        st.stop()
    img = Image.open(up).convert("RGB")
else:
    cam = st.camera_input("Take a picture")
    if not cam:
        st.info("Please snap a picture with your webcam.")
        st.stop()
    img = Image.open(cam).convert("RGB")

# ─── 5) DETECT ALL FACES ─────────────────────────────────────────
boxes, confidences = mtcnn_detect.detect(img)
n_faces = 0 if boxes is None else len(boxes)
if n_faces == 0:
    st.error("😞 No faces detected. Try a clearer photo or different lighting.")
    st.image(img, use_column_width=True)
    st.stop()

st.success(f"Detected **{n_faces}** face{'s' if n_faces>1 else ''} in your image.")

# ─── 6) ANNOTATE & SUMMARIZE ────────────────────────────────────
canvas    = img.copy()
draw      = ImageDraw.Draw(canvas)
font      = ImageFont.load_default()
summaries = []

for i, box in enumerate(boxes, start=1):
    x1, y1, x2, y2 = map(int, box)
    patch = img.crop((x1, y1, x2, y2))

    # 6.1 Align (catch any failures)
    try:
        face_tensor = mtcnn_align(patch)
    except Exception:
        face_tensor = None
    if face_tensor is None:
        summaries.append(f"• Face #{i}: alignment failed")
        continue
    if face_tensor.ndim == 4:
        face_tensor = face_tensor.squeeze(0)

    # 6.2 Embed
    with torch.no_grad():
        emb = resnet(face_tensor.unsqueeze(0).to(device))  # (1,512)
    emb_np = emb[0].cpu().numpy()[None, :]

    # 6.3 KNN lookup
    dists, idxs = knn.kneighbors(emb_np, n_neighbors=TOP_K)
    best_d     = float(dists[0][0])
    emb_index  = int(idxs[0][0])

    # 6.4 Decide name
    if best_d > UNKNOWN_THRESHOLD:
        name = "Unknown"
    else:
        class_id = labels[emb_index]
        name     = subjects[class_id]

    # 6.5 Draw box + label
    draw.rectangle([(x1, y1), (x2, y2)], outline="lime", width=2)
    draw.text((x1, y1-12), f"{name} ({best_d:.2f})", fill="red", font=font)

    summaries.append(f"• Face #{i}: `{name}` (dist={best_d:.2f}), box=({x1},{y1},{x2},{y2})")

# ─── 7) SHOW ORIGINAL + ANNOTATED SIDE-BY-SIDE ──────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original upload")
    st.image(img, use_column_width=True)
with col2:
    st.subheader("🖼️ Annotated")
    st.image(canvas, use_column_width=True)

# ─── 8) FACE-BY-FACE SUMMARY ─────────────────────────────────────
st.subheader("Face-by-face summary")
for line in summaries:
    st.text(line)

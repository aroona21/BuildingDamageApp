# 🏚️ Building Damage Assessment

An AI-powered **Building Damage Assessment System** built using Streamlit and a Siamese ResNet50 U-Net architecture. The application compares pre-disaster and post-disaster satellite images to identify and visualize building damage.

The system generates pixel-level damage segmentation masks, prediction overlays, and damage-class percentages to support post-disaster assessment.

---

## 📌 Project Overview

Building Damage Assessment uses deep learning and satellite imagery to analyze the impact of natural disasters on buildings.

Users can select sample disaster images or upload their own pre-disaster and post-disaster satellite images. The trained model processes the image pair and predicts the damage category for each pixel.

### Key Features

- Compare pre-disaster and post-disaster satellite images.
- Use provided sample disaster image pairs.
- Upload custom pre-disaster and post-disaster images.
- Perform pixel-level damage segmentation.
- Generate color-coded damage prediction masks.
- Display prediction overlays on post-disaster imagery.
- View predicted pixel-area distribution for each damage class.
- Download prediction masks, overlays, and original images.

---

## 🧠 Model Architecture

The project uses a **Siamese ResNet50 U-Net** architecture for image-based damage assessment.

The model processes two images:

1. Pre-disaster satellite image.
2. Post-disaster satellite image.

It extracts features from both images using a shared ResNet50 encoder. The absolute difference between corresponding feature maps is calculated, and the features are fused before being passed through the U-Net decoder.

The segmentation head produces predictions for five damage classes.

**Model details:**
- Architecture: Siamese U-Net
- Encoder: ResNet50
- Framework: PyTorch
- Number of output classes: 5
- Input: Pre-disaster and post-disaster RGB images
- Inference: 512 × 512 pixel tiles with 128-pixel overlap

---

## 🎨 Damage Classification

The model predicts five classes:

| Class | Damage Category | Color |
|---|---|---|
| 0 | Background | Black |
| 1 | No Damage | Green |
| 2 | Minor Damage | Yellow |
| 3 | Major Damage | Orange |
| 4 | Destroyed | Red |

---

## 📂 Project Structure

```text
BuildingDamageApp/
│
├── app.py
├── model_def.py
├── requirements.txt
├── README.md
│
├── checkpoints/
│   └── resnet50_damage_best.pt
│
└── sample_images/
    ├── Hurricane_Florence/
    │   ├── pre.jpg
    │   ├── post.jpg
    │   └── ground_truth.png
    │
    ├── Midwest_Flooding/
    │   ├── pre.png
    │   ├── post.png
    │   └── ground_truth.png
    │
    ├── Midwest_Flooding_2/
    ├── Santa_Rosa_Wildfire/
    └── SoCal_Fire/
```

---

## 💾 Model Checkpoint Download

The trained model checkpoint is approximately **188 MB** and is hosted separately on Google Drive because GitHub does not support regular Git uploads of individual files larger than 100 MB.

**Download the trained model:**

[Download resnet50_damage_best.pt from Google Drive](https://drive.google.com/file/d/1w1X9jWFwfbKttKSA-n3hAiA7OJZIW3FC/view?usp=sharing)

### Checkpoint Setup

1. Download `resnet50_damage_best.pt` from the Google Drive link above.
2. Open the downloaded project folder.
3. Create a folder named `checkpoints` if it does not already exist.
4. Place the downloaded checkpoint inside that folder.

The final path should be:

```text
BuildingDamageApp/
└── checkpoints/
    └── resnet50_damage_best.pt
```

**Important:** The application requires this checkpoint to perform damage predictions. The model architecture alone cannot produce predictions using the trained weights.

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/aroona21/BuildingDamageApp.git
```

### 2. Open the Project Folder

```bash
cd BuildingDamageApp
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the Model Checkpoint

Download the checkpoint from the Google Drive link above and place it inside the `checkpoints` folder.

---

## 🚀 Running the Application

After completing the installation and checkpoint setup, run:

```bash
streamlit run app.py
```

The application will open in your default browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

---

## 🖥️ How to Use the Application

### 1. Choose Image Source

The application provides two options:

- **Sample Images:** Select from the available disaster image pairs.
- **Upload Your Own:** Upload pre-disaster and post-disaster satellite images.

### 2. Select or Upload Images

For custom uploads, provide both images. Their dimensions must match.

### 3. Analyze Building Damage

Click the **Analyze Building Damage** button to run the model.

The application processes the images using sliding-window inference and generates the predicted damage mask.

### 4. View Results

The results include:

- Ground Truth Mask (when available for a sample).
- Predicted Damage Mask.
- Prediction Overlay.
- Damage Class Legend.
- Predicted Pixel-Area Distribution.

### 5. Download Results

Users can download:

- Pre-disaster image.
- Post-disaster image.
- Predicted damage mask.
- Color-coded prediction overlay.

---

## 📊 Prediction Method

The application uses sliding-window inference to process satellite images.

- Tile size: 512 × 512 pixels.
- Overlap: 128 pixels.
- Overlapping predictions are averaged to produce the final segmentation mask.
- Each pixel is assigned to the class with the highest predicted score.

The application also calculates the percentage of image pixels belonging to each damage class.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- Segmentation Models PyTorch
- ResNet50
- U-Net
- NumPy
- Pillow
- Streamlit

---

## 📦 Requirements

The required Python packages are listed in `requirements.txt`:

```text
streamlit>=1.35
torch>=2.2
torchvision>=0.17
segmentation-models-pytorch>=0.3.3
numpy>=1.24
Pillow>=10
```

Python 3.10 or 3.11 is recommended.

---

## ⚠️ Important Notes and Limitations

- Both input images must have identical dimensions.
- The trained checkpoint must be present in the `checkpoints` folder.
- Ground truth masks are available only for the provided sample images when included.
- Uploaded images do not have ground truth masks.
- Damage percentages represent pixel-area distribution, not verified building counts.
- Model predictions are decision-support outputs and should be reviewed before being used for real-world decisions.

---

## 👩‍💻 Author

**Aroona Noor**

GitHub: [@aroona21](https://github.com/aroona21)

---

## 📜 License

Add a license to this repository if you intend to distribute the project under specific reuse permissions.

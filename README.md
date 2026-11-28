# 🏚️ Building Damage Assessment

An AI-powered Building Damage Assessment System built using **Streamlit** and a **Siamese ResNet50 U-Net** architecture. The application compares pre-disaster and post-disaster satellite images to identify and visualize building damage.

The system generates pixel-level damage segmentation masks, prediction overlays, and damage-class percentages to support post-disaster assessment.

---

# 1. Project Information

**Project Title:** Building Damage Assessment

**Course:** Artificial Intelligence (AI 5C)

**University:** Bahria University, Islamabad

**Instructor:** Ms. Afrah Naeem

**Team Members:**

| Name | Roll Number |
|---|---|
| Aroona Noor | 01-134242-027 |
| Zahra Khan | 01-134242-139 |

**GitHub Repository:**  
https://github.com/aroona21/BuildingDamageApp

---

# 2. What the Project Does

Building Damage Assessment uses deep learning and satellite imagery to analyze the impact of natural disasters on buildings.

Users can select sample disaster image pairs or upload their own pre-disaster and post-disaster satellite images. The trained model processes both images and predicts the damage category for each pixel.

### Key Features

- Compare pre-disaster and post-disaster satellite images.
- Select from provided sample disaster image pairs.
- Upload custom pre-disaster and post-disaster images.
- Perform pixel-level damage segmentation.
- Generate color-coded damage prediction masks.
- Display prediction overlays on post-disaster imagery.
- View predicted pixel-area distribution for each damage class.
- Download prediction masks, overlays, and original images.

## Model Architecture

The project uses a **Siamese ResNet50 U-Net** architecture.

The model processes two images:

1. Pre-disaster satellite image.
2. Post-disaster satellite image.

It extracts features from both images using a shared ResNet50 encoder. The absolute difference between corresponding feature maps is calculated, and the features are fused before being passed through the U-Net decoder.

The segmentation head produces predictions for five damage classes.

| Model Component | Description |
|---|---|
| Architecture | Siamese U-Net |
| Encoder | ResNet50 |
| Framework | PyTorch |
| Output Classes | 5 |
| Input | Pre-disaster and post-disaster RGB images |
| Tile Size | 512 × 512 pixels |
| Tile Overlap | 128 pixels |

### Damage Classification

| Class | Damage Category | Color |
|---|---|---|
| 0 | Background | Black |
| 1 | No Damage | Green |
| 2 | Minor Damage | Yellow |
| 3 | Major Damage | Orange |
| 4 | Destroyed | Red |

---

# 3. What to Install

### Prerequisites

Before running the project, install:

- Python 3.10 or Python 3.11
- Git
- pip

### Required Python Libraries

All required libraries are listed in `requirements.txt`.

```text
streamlit>=1.35
torch>=2.2
torchvision>=0.17
segmentation-models-pytorch>=0.3.3
numpy>=1.24
Pillow>=10
```

Install the libraries using:

```bash
pip install -r requirements.txt
```

### Trained Model Checkpoint

The trained model checkpoint is approximately 188 MB. It is hosted separately on Google Drive because it exceeds GitHub's regular 100 MB individual file upload limit.

**Download the trained model:**

[Download resnet50_damage_best.pt from Google Drive](https://drive.google.com/file/d/1w1X9jWFwfbKttKSA-n3hAiA7OJZIW3FC/view?usp=sharing)

### Checkpoint Setup

1. Download `resnet50_damage_best.pt` using the Google Drive link above.
2. Open the downloaded project folder.
3. Create a folder named `checkpoints` if it does not already exist.
4. Place the downloaded checkpoint inside the `checkpoints` folder.

The final path should be:

```text
BuildingDamageApp/
└── checkpoints/
    └── resnet50_damage_best.pt
```

**Important:** The application requires this checkpoint to perform damage predictions. The model architecture alone cannot produce predictions without the trained weights.

### Dataset Information

The project uses satellite imagery from the xBD (xView2 Building Damage Assessment) dataset.

The provided sample disaster images are included in the `sample_images` folder for testing the application. The complete dataset is not included because of its large size.

If the complete dataset is required, it can be obtained from:

[INSERT DATASET DOWNLOAD LINK HERE]

---

# 4. How to Run the Project

Follow these steps in the given order to run the application on your computer.

### Step 1: Clone the GitHub Repository

Open a terminal or command prompt and run:

```bash
git clone https://github.com/aroona21/BuildingDamageApp.git
```

### Step 2: Open the Project Folder

```bash
cd BuildingDamageApp
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### Step 5: Install the Required Libraries

```bash
pip install -r requirements.txt
```

### Step 6: Download and Set Up the Model Checkpoint

Download `resnet50_damage_best.pt` from the Google Drive link provided in Section 3.

Place it inside the `checkpoints` folder.

Make sure the final file path is:

```text
checkpoints/resnet50_damage_best.pt
```

### Step 7: Run the Application

Run the following command from the main project folder:

```bash
streamlit run app.py
```

The application will open in your default web browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

---

# 5. Expected Output and How to Use the Application

When the application runs successfully, the TerraScan Building Damage Assessment interface will appear in the browser.

### Step 1: Choose Image Source

The application provides two options:

- **Sample Images:** Select from the available disaster image pairs.
- **Upload Your Own:** Upload custom pre-disaster and post-disaster satellite images.

### Step 2: Select or Upload Images

For custom uploads, provide both pre-disaster and post-disaster images. Their dimensions must match.

### Step 3: Analyze Building Damage

Click the **Analyze Building Damage** button to run the trained model.

The application processes the images using sliding-window inference and generates a predicted damage mask.

### Step 4: View Results

The application displays:

- Ground Truth Mask (when available for a sample).
- Predicted Damage Mask.
- Prediction Overlay.
- Damage Class Legend.
- Predicted Pixel-Area Distribution.

### Step 5: Download Results

Users can download:

- Pre-disaster image.
- Post-disaster image.
- Predicted damage mask.
- Color-coded prediction overlay.

### Expected Results

When the application works correctly, users should see the original satellite images alongside the predicted damage mask and overlay.

The application also displays the percentage distribution of pixels belonging to each damage class.

**Application Screenshot:**

[INSERT A SCREENSHOT OF YOUR RUNNING STREAMLIT APPLICATION HERE]

---

# 6. File and Folder Descriptions

The project contains the following files and folders:

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

| File / Folder | Description |
|---|---|
| `app.py` | Main Streamlit application. Handles image uploads, predictions, visualizations, and downloads. |
| `model_def.py` | Contains the Siamese ResNet50 U-Net model architecture. |
| `requirements.txt` | Contains the Python libraries required to run the project. |
| `README.md` | Provides project information and instructions for installation and execution. |
| `checkpoints/` | Stores the trained model checkpoint. |
| `resnet50_damage_best.pt` | Contains the trained model weights used for predictions. |
| `sample_images/` | Contains sample pre-disaster and post-disaster satellite images and available ground truth masks. |

---

## Prediction Method

The application uses sliding-window inference to process satellite images.

| Setting | Value |
|---|---|
| Tile Size | 512 × 512 pixels |
| Overlap | 128 pixels |
| Prediction Method | Sliding-window inference |

Overlapping predictions are averaged to produce the final segmentation mask. Each pixel is assigned to the class with the highest predicted score.

The application calculates the percentage of image pixels belonging to each damage class.

---

## Important Notes and Limitations

- Both input images must have identical dimensions.
- The trained checkpoint must be present in the `checkpoints` folder.
- Ground truth masks are available only for provided sample images when included.
- Uploaded images do not have ground truth masks.
- Damage percentages represent pixel-area distribution, not verified building counts.
- Model predictions are decision-support outputs and should be reviewed before being used for real-world decisions.

---

## Technologies Used

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

## Author

**Aroona Noor**

**Zahra Khan**
GitHub: https://github.com/aroona21

---

## License

No specific open-source license has currently been assigned to this project.

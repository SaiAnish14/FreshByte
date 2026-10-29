# 🍎 FreshByte: AI-Powered Food Freshness Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/Model-MobileNetV2-009688?style=for-the-badge&logo=google&logoColor=white" alt="MobileNetV2" />
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Notebook-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License: MIT" />
</p>

<p align="center">
  <strong>FreshByte</strong> is an intelligent Computer Vision and Deep Learning system designed to accurately classify produce as <strong>Fresh</strong> or <strong>Rotten</strong> in real-time. Built using <strong>MobileNetV2 transfer learning</strong>, FreshByte features an intuitive <strong>Streamlit web application</strong>, live camera/webcam capture, confidence scores, probability gauge meters, and an end-to-end <strong>Google Colab</strong> training pipeline with Hugging Face integration.
</p>

---

## 📑 Table of Contents

- [🌟 Overview & Motivation](#-overview--motivation)
- [⚡ Key Features](#-key-features)
- [🍎 Supported Fruit Categories](#-supported-fruit-categories)
- [🧠 Deep Learning Architecture & Pipeline](#-deep-learning-architecture--pipeline)
- [📊 Dataset & Data Leakage Prevention](#-dataset--data-leakage-prevention)
- [📁 Repository Structure](#-repository-structure)
- [🚀 Quick Start & Installation](#-quick-start--installation)
  - [Prerequisites](#prerequisites)
  - [1. Local Setup](#1-local-setup)
  - [2. Google Colab Training Workflow](#2-google-colab-training-workflow)
- [🌐 Free Cloud Deployment (Streamlit Cloud)](#-free-cloud-deployment-streamlit-cloud)
- [📈 Evaluation Metrics & Performance](#-evaluation-metrics--performance)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [🎓 Course Project Details](#-course-project-details)
- [📜 License & References](#-license--references)

---

## 🌟 Overview & Motivation

Post-harvest food loss and microbial spoilage represent critical global economic, environmental, and food safety challenges:
- **Food Spoilage in Supply Chains:** Produce that begins to decay produces ethylene gas and releases fungal spores, accelerating decay in neighboring produce and contaminating entire crates.
- **Subjective Human Inspection:** Traditional manual quality checks rely on human visual inspection, which is slow, labor-intensive, fatigue-prone, and inconsistent across different observers and lighting conditions.
- **Expensive Hardware Alternatives:** Specialized systems such as hyperspectral imaging, electronic noses, and chemical sensors offer high precision but require expensive, proprietary equipment that is inaccessible to everyday consumers, small-scale grocers, and community food pantries.

**FreshByte solves this problem** by providing a lightweight, accessible, AI-powered computer vision pipeline. Using a smartphone camera or uploaded photograph, users can instantly determine whether produce is fresh or spoiled with high statistical confidence.

---

## ⚡ Key Features

- **📸 Dual Input Modes:**
  - **File Upload:** Drag-and-drop or upload high-resolution photos (`.jpg`, `.jpeg`, `.png`).
  - **Live Camera Capture:** Capture photos directly from a laptop webcam or mobile smartphone camera using Streamlit's native `st.camera_input`.
- **🧠 State-of-the-Art Transfer Learning:** Employs **MobileNetV2** pre-trained on 1.4 million ImageNet samples, utilizing inverted residual blocks and depthwise separable convolutions for high accuracy with minimal computational overhead.
- **⚡ Ultra-Lightweight (~10 MB):** The entire model file is approximately 10 MB, enabling rapid load times and fast inference even on budget CPU servers without requiring expensive GPUs.
- **📊 Real-Time Confidence Gauges:** Delivers both the final classification label and granular probability breakdown meters for both Fresh and Rotten classes.
- **⚠️ Low-Confidence Safety Safeguards:** Automatically alerts the user if confidence drops below 70%, flagging unclear lighting, occluded views, or out-of-distribution fruit types.
- **🔄 Two-Phase Progressive Fine-Tuning:**
  - *Phase 1:* Frozen backbone with a high learning rate (`1e-3`) to train the classification head.
  - *Phase 2:* Top 30 backbone layers unfrozen with a low learning rate (`1e-5`) for precise domain adaptation while preventing catastrophic forgetting.
- **🛡️ Strict Anti-Leakage Protocol:** Stratified train/validation/test split performed *prior* to data augmentation, ensuring no synthetic or augmented variants cross split boundaries.
- **💻 Dual User Interfaces:** Includes an interactive web app powered by **Streamlit** and a **Gradio** web preview embedded inside the Google Colab notebook.

---

## 🍎 Supported Fruit Categories

The model was trained on **8 popular fruit types** containing both pristine fresh specimens and real-world rot variations (spots, fungal growth, discoloration, wrinkled skin, bruising):

| Fruit | Fresh Indicators | Rotten Indicators |
|---|---|---|
| 🍏 **Apple** | Firm peel, uniform coloration, smooth skin | Soft spots, browning, microbial decay |
| 🍌 **Banana** | Bright yellow peel, intact stem | Dark sunken patches, extensive blackening, mushiness |
| 🍊 **Orange** | Vibrant citrus peel, firm texture | Mold spores, fungal patches, soft collapse |
| 🍇 **Grape** | Plump turgor, bloom intact, firm clusters | Wrinkled skin, juice leakage, mold clusters |
| 🍈 **Guava** | Crisp light green/yellow skin, firm body | Deep brown lesions, soft bruising, rot spots |
| 🫐 **Jujube** | Smooth glossy reddish-brown skin | Severe shriveling, localized decay, darkening |
| 🍎 **Pomegranate** | Leathery taut crimson rind, heavy crown | Discolored blotches, cracked soft rind, fungal spots |
| 🍓 **Strawberry** | Vivid red luster, firm seed structure | Gray mold (*Botrytis*), collapse, discoloration |

---

## 🧠 Deep Learning Architecture & Pipeline

FreshByte leverages transfer learning with **MobileNetV2**, an efficient convolutional neural network architecture tailored for edge and mobile vision tasks.

```
                  ┌──────────────────────────────┐
                  │   Input Photo (JPG / PNG)    │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ Image Preprocessing Pipeline │
                  │  - EXIF Auto-Orientation     │
                  │  - Bilinear Resize (224x224) │
                  │  - Rescaling to [-1, 1]      │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │     MobileNetV2 Backbone     │
                  │  (ImageNet Pre-trained Base) │
                  │  - Inverted Residual Blocks  │
                  │  - Depthwise Separable Convs │
                  │  - Linear Bottlenecks        │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │  GlobalAveragePooling2D()    │
                  │   (Spatial Dimensionality)   │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │        Dropout (0.2)         │
                  │   (Regularization Layer)     │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │   Dense(1, Sigmoid Output)   │
                  │   P(Rotten) ∈ [0.0, 1.0]     │
                  └──────────────┬───────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
        P(Rotten) < 0.50            P(Rotten) ≥ 0.50
        ┌───────────────────┐       ┌───────────────────┐
        │  ✅ Class: FRESH  │       │  🚫 Class: ROTTEN │
        └───────────────────┘       └───────────────────┘
```

### Architectural Specifications

- **Input Dimension:** $224 \times 224 \times 3$ RGB.
- **Normalization:** Scaled to $[-1, 1]$ via MobileNetV2 `preprocess_input` (embedded directly within the computational graph).
- **Pooling Layer:** `GlobalAveragePooling2D()` converts the $7 \times 7 \times 1280$ feature maps into a 1280-dimensional feature vector.
- **Dropout Rate:** $0.20$ to prevent co-adaptation of features.
- **Output Layer:** Single neuron with Sigmoid activation function outputting $P(\text{Rotten})$.
- **Loss Function:** Binary Cross-Entropy:
  $$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \Big[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \Big]$$

### Two-Phase Training Schedule

| Parameter | Phase 1 (Feature Extraction) | Phase 2 (Fine-Tuning) |
|---|---|---|
| **Trainable Layers** | Top custom classifier head only (Backbone frozen) | Top 30 MobileNetV2 layers + Classifier head |
| **Optimizer** | Adam ($\beta_1 = 0.9, \beta_2 = 0.999$) | Adam ($\beta_1 = 0.9, \beta_2 = 0.999$) |
| **Learning Rate** | $1 \times 10^{-3}$ ($0.001$) | $1 \times 10^{-5}$ ($0.00001$) |
| **Max Epochs** | 10 epochs | 15 epochs |
| **Batch Size** | 32 | 32 |
| **Callbacks** | `EarlyStopping(patience=5)`, `ModelCheckpoint` | `EarlyStopping(patience=5)`, `ReduceLROnPlateau` |

---

## 📊 Dataset & Data Leakage Prevention

FreshByte uses the peer-reviewed **Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality** (*Sultana, Jahan & Uddin, 2022, Data in Brief*), sourced directly from Hugging Face:
- **Hugging Face Repository:** [`Project-AgML/fresh_rotten_fruit_classification`](https://huggingface.co/datasets/Project-AgML/fresh_rotten_fruit_classification)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Subset Selected:** `raw` subset containing **3,200 pristine original photos** (1,600 Fresh + 1,600 Rotten).
- **Balance:** Exactly 50% Fresh and 50% Rotten across all 8 fruit categories.

### Strict Anti-Leakage Protocol
Many published models mistakenly augment the entire dataset *before* splitting, causing synthetic duplicates of test images to contaminate the training split. FreshByte strictly implements **stratified splitting prior to augmentation**:

```
                       3,200 Raw Images
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
           Training       Validation        Test
            (70%)           (15%)          (15%)
         2,240 Images     480 Images     480 Images
               │              │              │
        [Augmentation]     [Clean]        [Clean]
        - Random Flip     No Augment     No Augment
        - Rotation ±15°
        - Zoom ±10%
        - Contrast Jitter
```

---

## 📁 Repository Structure

```
FreshByte/
│
├── .streamlit/
│   └── config.toml               # Streamlit theme branding (Primary green, layout config)
│
├── model/
│   ├── README.md                 # Instructions for placing trained model weights
│   ├── food_freshness_model.keras# Trained MobileNetV2 Keras model (~10 MB, generated via notebook)
│   └── metrics.json              # Evaluated test metrics for dynamic sidebar display
│
├── notebooks/
│   └── Food_Freshness_Detection.ipynb # Complete Google Colab end-to-end training notebook
│
├── report/
│   └── Food_Freshness_Detection_Project_Record.docx # Full course academic project documentation
│
├── .gitignore                    # Git exclusions (virtualenvs, cache, checkpoints)
├── app.py                        # Streamlit web application frontend & inference engine
├── LICENSE                       # MIT License
├── README.md                     # Comprehensive project documentation
└── requirements.txt              # Production Python package dependencies
```

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python **3.10, 3.11, or 3.12** installed on your system.
- `pip` package manager.
- A modern web browser (Chrome, Firefox, Safari, Edge).

### 1. Local Setup

Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/SaiAnish14/FreshByte.git
cd FreshByte

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (CMD):
.venv\Scripts\activate.bat
# On macOS / Linux:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

# Run the Streamlit web application
streamlit run app.py
```

The app will automatically launch in your browser at:
👉 **`http://localhost:8501`**

---

### 2. Google Colab Training Workflow

You can train or fine-tune the model with free GPU acceleration directly on Google Colab **without needing to manually download or upload any dataset files**:

1. Open [Google Colab](https://colab.research.google.com).
2. Click **File** ➔ **Upload notebook**, and upload `notebooks/Food_Freshness_Detection.ipynb`.
3. In the Colab menu, select:
   **Runtime** ➔ **Change runtime type** ➔ **T4 GPU** ➔ **Save**.
4. Click **Runtime** ➔ **Run all** (or execute cells sequentially).
5. The notebook will:
   - Automatically download the dataset from Hugging Face Hub.
   - Perform stratified train/validation/test splitting.
   - Run Phase 1 training (head training) and Phase 2 fine-tuning.
   - Generate confusion matrices, ROC curves, and classification metrics.
   - Launch an inline **Gradio** web app.
   - Package `food_freshness_model.keras` and `metrics.json` into a downloadable `project_outputs.zip`.
6. Unzip `project_outputs.zip` and place the two files into your local `model/` directory.

---

## 🌐 Free Cloud Deployment (Streamlit Cloud)

You can deploy FreshByte live on the web for free using **Streamlit Community Cloud**:

1. Fork or push this repository to your GitHub account (`SaiAnish14/FreshByte`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
3. Click **Create app** (or **New app**).
4. Configure the deployment settings:
   - **Repository:** `SaiAnish14/FreshByte`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Advanced settings**:
   - Set **Python version:** `3.11` *(Recommended for optimal TensorFlow compatibility)*.
6. Click **Deploy!**
7. Within 2-3 minutes, your live web application will be accessible worldwide via a custom `.streamlit.app` URL.

---

## 📈 Evaluation Metrics & Performance

Model performance is rigorously measured on the independent, unseen test split ($N = 480$ images):

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Why Rotten Recall is the Priority Metric
In food freshness detection, **false negatives** (predicting a rotten fruit as fresh) pose food safety hazards and cause fungal decay to spread across storage batches. Conversely, a false positive simply prompts manual human verification. Therefore, the fine-tuning optimization emphasizes high **Recall for the Rotten class** while maintaining high overall accuracy.

When `model/metrics.json` is generated, the Streamlit app sidebar dynamically parses and renders the official test accuracy, precision, recall, and F1 metrics.

---

## 🛠️ Troubleshooting

| Issue / Error | Probable Cause | Recommended Resolution |
|---|---|---|
| **"No trained model found"** | `model/food_freshness_model.keras` has not been placed in the folder. | Train the model using the notebook or upload the `.keras` file directly through the app's upload interface. |
| **"Model could not be loaded"** | Discrepancy between the TensorFlow version used during training and inference. | In `requirements.txt`, pin `tensorflow-cpu` to the exact version printed in cell 15 of the training notebook. |
| **"Low confidence warning"** | The uploaded image is blurry, poorly lit, or contains an unsupported fruit type. | Ensure the fruit is centered, well-lit, and belongs to one of the 8 trained categories. |
| **Streamlit Cloud Build Error** | Selected an unsupported Python version (e.g. Python 3.13). | In Streamlit Cloud **Advanced settings**, set the Python runtime explicitly to **Python 3.11**. |

---

## 🎓 Course Project Details

- **Institution:** Prasad V. Potluri Siddhartha Institute of Technology (PVPSIT)
- **Department:** Department of Computer Science & Engineering
- **Academic Year:** 2026-2027
- **Course:** Deep Learning Laboratory / Course Project
- **Project Title:** Food Freshness Detection Using Deep Learning (MobileNetV2 Transfer Learning)
- **Faculty Guide / Mentor:** Ms. Y. Surekha
- **Lead Developer & Author:** [Sai Anish](https://github.com/SaiAnish14)

---

## 📜 License & References

### License
This project is open-source and released under the [MIT License](LICENSE).

### Academic Citations
1. **Dataset:**
   > Sultana, N., Jahan, M., & Uddin, M. S. (2022). *An extensive dataset for successful recognition of fresh and rotten fruits.* **Data in Brief**, 44, 108552. DOI: [10.1016/j.dib.2022.108552](https://doi.org/10.1016/j.dib.2022.108552)
2. **Backbone Architecture:**
   > Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)**, pp. 4510-4520.
3. **Streamlit Framework:**
   > Treveil, M., et al. (2020). *Streamlit: An open-source app framework for Machine Learning and Data Science.*

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/SaiAnish14">Sai Anish</a>
</p>

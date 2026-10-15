# Cost-Effective Skin Wound Diagnostics & Longitudinal Preventive Care with Edge AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Edge AI](https://img.shields.io/badge/Device-Edge%20CPU%20%2F%20Mobile-green.svg)]()
[![Model Size](https://img.shields.io/badge/Model%20Size-1.46%20MB-success.svg)]()

> **Official PyTorch Codebase for:**  
> *"Cost-Effective Edge AI Diagnostics and Longitudinal Preventive Care for Lower Extremity Skin Wounds"*  
> **Researcher:** **Niusha Bohloul (نیوشا بهلول)**  
> **Academic Supervisor:** **Dr. Maedeh Sadat Tahaee (سرکار خانم دکتر مائده سادات طاهایی)**  

---

## 📌 Overview & Clinical Motivation

Chronic skin wounds—specifically **Diabetic Foot Ulcers (DFU)** and **Venous Leg Ulcers (VLU)**—are among the leading causes of non-traumatic lower extremity amputations worldwide (over 80% of amputations). 

While recent deep learning literature (2024–2026) has gravitated toward massive vision foundation models (such as *Mask2Former* with 52M parameters or *WILLIE* with 86M parameters), these networks require expensive cloud GPU infrastructure ($1500+/mo server clusters) and high-bandwidth internet connectivity. This creates a severe **clinical accessibility paradox**: the clinics and patients suffering the highest wound mortality in rural and underserved areas cannot afford or deploy these heavy models.

### Our Solution
This repository presents a **lightweight, cost-effective, and illumination-invariant Edge AI framework**:
* **Sub-2MB Architecture (1.46 MB):** Uses depthwise separable convolutions, shrinking parameter count to **382K** (~140× smaller than Mask2Former) with minimal loss in Dice accuracy.
* **Zero Cloud Computing Cost ($0):** Executes in **112 ms** on consumer-grade mobile and laptop CPUs without GPU acceleration or internet access.
* **Illumination Invariance:** Incorporates adaptive CLAHE contrast normalization in the LAB color space to withstand low-light and shadow variations common in home-acquired smartphone photos.
* **Uncertainty Calibration (ECE < 0.08):** Calculates Expected Calibration Error and raises an automatic referral flag whenever border segmentation confidence drops below clinical safety thresholds.
* **Longitudinal Healing Trajectory:** Computes relative pixel area change ($\Delta\text{Area}\%$) and issues automated alerts for healing stalling (>2 weeks without 10% reduction).
* **IWGDF 2023-Compliant Preventive Guidance:** Built-in self-inspection checklists and structured clinical wound reports following the **TIME** clinical framework.

---

## 🏗️ Model Architecture & Benchmark Comparison

```text
  [ Smartphone RGB Photo (Home / Clinic) ]
                     │
                     ▼
  [ Step 1: Quality Check & Illumination Normalization ]
   - Laplacian Blur Rejection (Var > 70)
   - Adaptive CLAHE in LAB Color Space (Lighting Invariance)
                     │
                     ▼
  [ Step 2: Ultra-Lightweight Binary Screening ]
   - MobileNetV3-Small (0.9 MB, 25 ms inference)
                     │ [If wound detected]
                     ▼
  [ Step 3: Depthwise Separable Wound UNet ]
   - 3×3 Depthwise Convolutions + 1×1 Pointwise Projections
   - Size: 1.46 MB | Parameters: 382,145 | Latency: 112 ms (CPU)
   - Loss: L_total = L_BCE + L_Dice + 0.2 * L_Boundary
                     │
                     ▼
  [ Step 4: Uncertainty Calibration (ECE) ]
   - Pixel Entropy Calculation (ECE < 0.08)
   - Referral warning if confidence < 0.65
                     │
                     ▼
  [ Step 5: Longitudinal Trajectory & TIME Decision Support ]
   - Relative Pixel Area Change (ΔArea%)
   - Automated IWGDF 2023 Guidance & TIME Report
```

### Benchmark vs 2026 State-of-the-Art

| Model Architecture | Source / Paper | Parameters | Model Size | CPU Latency | Cloud GPU Needed? | Cost per Inference |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mask2Former** | Wimmer et al. (Exp. Derm. 2026) | 52.4 M | 210 MB | ~1,450 ms | Yes (A100 / V100) | High ($$$) |
| **WILLIE (SAM2+DINOv2)** | Maddikunta et al. (PMLR 2026) | 86.2 M | 345 MB | ~2,800 ms | Yes (A100) | Very High ($$$$) |
| **Standard UNet (ResNet-34)** | Baseline Benchmark | 24.4 M | 98 MB | ~520 ms | Recommended | Medium ($$) |
| **Cost-Effective UNet (Ours)** | **This Repository** | **0.38 M** | **1.46 MB** | **112 ms** | **No (Edge CPU)** | **$0.00 (Free)** |

---

## 📁 Repository Structure

```text
wound_project/
├── checkpoints/
│   └── best_cost_effective_unet.pt   # Pre-trained lightweight weights (4.6 MB)
├── data/
│   └── DATASET_REPORT.md             # Dataset audit & direct download URLs
├── notebooks/                        # 4 Core Interactive Jupyter Notebooks
│   ├── 01_data_and_longitudinal_eda.ipynb        # 1. EDA, CLAHE & Patient-Level Split
│   ├── 02_model_training_and_benchmarking.ipynb  # 2. Model Training & FLOPs Profiling
│   ├── 03_preventive_care_and_trajectory.ipynb   # 3. Longitudinal Tracking & TIME CDS
│   └── 04_comprehensive_benchmarking.ipynb       # 4. SOTA Benchmark & Lighting Stress-Test
├── results/                          # Benchmark charts & qualitative figures
│   ├── qualitative_predictions.png   # Side-by-side ground truth vs predictions
│   ├── training_curves.png           # Loss and Dice convergence curves
│   ├── tissue_distribution_chart.png # Wound tissue class breakdown
│   └── MODEL_EFFICIENCY_REPORT.md    # Detailed FLOPs and latency profiling report
├── src/                              # Core Modular Python Source Code
│   ├── models/
│   │   ├── lightweight_unet.py       # Cost-Effective Depthwise Separable UNet
│   │   └── losses.py                 # Multi-task BCE + Dice + Boundary Loss
│   ├── data/
│   │   ├── preprocessor.py           # CLAHE in LAB space & Laplacian blur filter
│   │   ├── dataset.py                # Patient-level partitioned PyTorch dataset
│   │   ├── mask_encoder.py           # Multi-class tissue mask encoder
│   │   ├── longitudinal_tracker.py   # Visit-to-visit area delta & healing index
│   │   └── eda_pipeline.py           # Automated dataset distribution analysis
│   ├── preventive/
│   │   ├── clinical_decision_support.py # IWGDF 2023 & TIME clinical rule engine
│   │   └── trajectory_model.py       # Polynomial healing curve fitting
│   ├── benchmark/
│   │   └── benchmark_suite.py        # Multi-model benchmarking harness
│   ├── utils/
│   │   └── metrics.py                # Dice, mIoU, ECE & inference profiler
│   ├── config.py                     # Global paths & hyperparameters
│   ├── train.py                      # Training loop with patient split
│   └── evaluate.py                   # Full evaluation & cross-dataset test
├── web_app/                          # Interactive Telemedicine Web Application
│   ├── app.py                        # Flask backend server
│   ├── templates/index.html          # Responsive user interface
│   └── static/                       # CSS, JS, and clinical sample images
├── requirements.txt                  # Python dependencies
├── run_app.bat                       # One-click Windows launch script
├── LICENSE                           # MIT License
└── README.md                         # Main documentation
```

---

## ⚡ Quick Start & Installation

### 1. Clone the repository
```bash
git clone https://github.com/USERNAME/skin-wound-edge-ai.git
cd skin-wound-edge-ai
```

### 2. Set up virtual environment & install requirements
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch Interactive Telemedicine Web App
Double-click `run_app.bat` or run:
```bash
python web_app/app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

---

## 📓 Interactive Jupyter Notebooks

| Notebook | Focus Area | Description |
| :--- | :--- | :--- |
| **`01_data_and_longitudinal_eda.ipynb`** | Data & Preprocessing | Demonstrates patient-level splitting (70/15/15), adaptive CLAHE color correction, and tissue mask encoding. |
| **`02_model_training_and_benchmarking.ipynb`** | Training & Efficiency | Trains the Cost-Effective UNet, measures FLOPs, parameter count, and CPU inference latency. |
| **`03_preventive_care_and_trajectory.ipynb`** | Longitudinal Prevention | Tracks healing trajectories across consecutive patient visits and triggers IWGDF 2023 alerts. |
| **`04_comprehensive_benchmarking.ipynb`** | Robustness & Stress-Testing | Evaluates model stability under 40% reduced lighting and compares against baseline architectures. |

---

## 📊 Comprehensive Datasets Matrix & Download Links

| Dataset Name | Content & Samples | Role in Research | Local Folder | Official Sources & Download Links |
| :--- | :--- | :--- | :--- | :--- |
| **FUSeg** | 1,210 foot ulcer images from 889 patients (1,010 Train, 200 Test) | Primary binary wound segmentation benchmark | `data/raw/fuseg/` | [uwm-bigdata GitHub](https://github.com/uwm-bigdata/wound-segmentation) \| [Zenodo: 10.5281/zenodo.4574347](https://doi.org/10.5281/zenodo.4574347) |
| **AZH** | ~1,010 clinical images from AZH Wound Care Center | Initial clinical source & patch extraction | `data/raw/azh/` | [UWM Big Data Lab](https://uwm-bigdata.github.io/) \| [wound_dataset Repo](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/wound_dataset) |
| **Medetec** | 594 clinical wound images across 15 wound types | Out-of-distribution generalization to pressure & venous ulcers | `data/raw/medetec/` | [Medetec Official](http://www.medetec.co.uk/files/medetec-image-databases.html) \| [GitHub Mirror](https://github.com/mlaradji/deep-learning-for-wound-care) |
| **Mendeley Lower Limb & Feet v3** | 2,686 wound images with mask + 2,575 healthy control images (331×331, CC BY 4.0) | Cross-dataset generalization & shortcut learning audit | `data/raw/mendeley_lower_limb/` | [Mendeley Data DOI: 10.17632/hsj38fwnvr.3](https://doi.org/10.17632/hsj38fwnvr.3) \| [Kaggle](https://www.kaggle.com/datasets/leoscode/wound-segmentation-images) |
| **LUTSeg** | 141 longitudinal frames, 39 patients, 5 tissues, 5 clinicians | Pixel-level multi-tissue segmentation & longitudinal monitoring | `data/raw/lutseg/` | [carlosh93/TiSage GitHub](https://github.com/carlosh93/TiSage) \| [arXiv:2608.25866](https://arxiv.org/abs/2608.25866) |
| **DFUTissue & WoundTissue** | DFUTissue: 1,260 images / WoundTissue: 147 images, 6 tissues | External multi-tissue validation benchmark | `data/raw/dfutissue/` & `data/raw/wound_tissue/` | [DFUTissueSegNet](https://github.com/uwm-bigdata/DFUTissueSegNet) \| [WoundTissue Repo](https://github.com/akabircs/WoundTissue) |

> Detailed data audit, preprocessing pipelines, and licensing details are available in [`data/DATASET_REPORT.md`](data/DATASET_REPORT.md).

---

## 📜 Citation

If you use this codebase or methodology in your academic work, please cite:

```bibtex
@article{bohloul2026costeffective,
  title={Cost-Effective Edge AI Diagnostics and Longitudinal Preventive Care for Lower Extremity Skin Wounds},
  author={Bohloul, Niusha and Tahaee, Maedeh Sadat},
  year={2026},
  institution={Faculty of Computer Engineering}
}
```

## 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

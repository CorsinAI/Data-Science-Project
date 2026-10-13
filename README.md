# Facial Emotion Recognition Pipeline

End-to-end system for detecting emotions in facial images using pretrained and custom-trained deep learning models, with an interactive dashboard and ensemble inference.

![Model Comparison](docs/images/overall_comparison.png)

## Overview

Recognizing human emotions from facial expressions is a well-studied yet challenging computer vision task with applications in human-computer interaction, accessibility, and behavioral analysis. This project builds a complete pipeline -- from data quality assessment through model training and hyperparameter tuning to interactive visualization -- that compares three detection approaches and combines them in an ensemble strategy.

Seven emotion classes are supported: **angry**, **disgusted**, **fearful**, **happy**, **neutral**, **sad**, and **surprised**.

## Team

This project was developed at **HSLU** as part of the **Data Science Project** course.

Members: Marco Blum, Kim Bortis, Lukas Pfründer, Corsin Hidber

## Data

### Faces Dataset (primary)

- **Source**: [TODO: link to original source]
- **Size**: 35,887 images (28,709 train / 7,178 test)
- **Format**: PNG, organized into 7 emotion folders per split
- **Access**: Not included in the repository. Download and place under `data/raw/Faces_Dataset/`.

### Emotion Dataset 3

- **Source**: [TODO: link to original source]
- **Size**: 15,339 images (12,271 train / 3,068 test)
- **Format**: 48x48 aligned JPEG faces with CSV label files
- **Class imbalance**: `happy` is the majority class at 38.8%; `disgusted` is severely underrepresented at 2.3%.
- **Access**: Not included in the repository. Download and place under `data/raw/emotion dataset 3/`.

### Faces Dataset 2 (supplementary)

- **Size**: 172 high-quality images with demographic metadata (gender, age, country)
- **Use**: Quality benchmarking and demographic bias analysis

## Approach

### Models

| Model | Type | Description |
|-------|------|-------------|
| **DeepFace-Emotion** | Pretrained | Emotion backend from the [DeepFace](https://github.com/serengil/deepface) library |
| **FER** | Pretrained | [Facial Expression Recognition](https://github.com/justinshenk/fer) CNN |
| **Custom-ResNet18** | Fine-tuned | ResNet18 (ImageNet-pretrained) with a custom classification head, trained on project data |

### Custom Model Design Decisions

- **Architecture**: ResNet18 chosen for its balance of accuracy and training speed on a consumer GPU.
- **Loss**: Cross-entropy with label smoothing (0.1) as baseline; focal loss explored for class imbalance.
- **Regularization**: Dropout (0.3), weight decay (0.05), MixUp augmentation.
- **Optimizer**: AdamW with OneCycleLR scheduling.
- **Class imbalance handling**: Inverse-log class weighting; weighted random sampling explored as an alternative.

### Ensemble Strategy

The ensemble uses **Custom-ResNet18 as the base predictor** and overrides its prediction when DeepFace and FER both agree on a different label. On the Faces Dataset test set, this produced a net gain of +17 correct predictions out of 6,684, raising accuracy from 67.59% to 67.85%.

## Training Setup

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet18 (pretrained on ImageNet) |
| Image size | 224x224 |
| Optimizer | AdamW |
| Learning rate | 3e-4 (baseline) |
| Batch size | 128 |
| Epochs | 30 (baseline) |
| Weight decay | 0.05 |
| Label smoothing | 0.1 |
| Hardware | Local GPU with 8 GB VRAM |
| Seed | 42 |

A systematic hyperparameter search across 12 configurations varied learning rate, batch size, epoch count, loss function, augmentation strategy, and class balancing method. See [`experiments/EXPERIMENTS_SUMMARY.md`](experiments/EXPERIMENTS_SUMMARY.md) for full details.

## Results

### Model Comparison on Faces Dataset Test Set

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|----------|-----------------|--------------|----------|
| DeepFace-Emotion | 53.13% | 51.57% | 47.46% | 48.76% |
| FER | 63.33% | 60.36% | 58.33% | 58.95% |
| Custom-ResNet18 | 67.59% | 66.29% | 62.87% | 63.97% |
| Ensemble | 67.85% | 67.54% | 62.40% | 63.86% |

### Best Custom-ResNet18 Checkpoint (`runs/exp2_clean/`)

- **Test accuracy**: 68.05%
- **Test macro F1**: 65.30%

### Hyperparameter Tuning Highlights

| Experiment | Test Accuracy | Test Macro F1 |
|------------|--------------|---------------|
| baseline_resnet18 | 66.85% | 63.68% |
| aug_heavy | 68.36% | 65.52% |
| loss_focal | 67.61% | 65.13% |
| lr_high_1e3 | 67.07% | 63.93% |
| epochs_50_extended | 67.35% | 64.58% |
| batch_64_smaller | 67.46% | 65.00% |

![Per-Emotion F1 Heatmap](docs/images/test_f1_heatmap.png)

![Model Agreement by Emotion](docs/images/agreement_by_emotion.png)

All metrics are taken directly from experiment logs and evaluation reports in the repository.

## How to Reproduce

### 1. Environment Setup

```bash
git clone <repository-url>
cd <repository-name>
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note**: `requirements.txt` pins PyTorch with CUDA 12.4 (`torch==2.6.0+cu124`). If your system uses a different CUDA version or CPU only, install PyTorch separately following [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

### 2. Data Download

Download the datasets and place them under `data/raw/` matching this structure:

```
data/raw/
  Faces_Dataset/
    train/{angry,disgusted,fearful,happy,neutral,sad,surprised}/
    test/{angry,disgusted,fearful,happy,neutral,sad,surprised}/
  emotion dataset 3/
    DATASET/train/ and DATASET/test/
    train_labels.csv and test_labels.csv
```

### 3. Training

```bash
# Single experiment (baseline)
python experiments/run_experiment.py experiments/configs/baseline.yaml \
    --data_root data/raw/Faces_Dataset

# All 12 experiments
python experiments/run_batch.py --data_root data/raw/Faces_Dataset
```

Outputs (weights, metrics, confusion matrices) are saved under `experiments/runs/<experiment_name>/`.

### 4. Evaluation

```bash
# Evaluate all three models on Faces Dataset
python examples/ensemble_faces_dataset.py

# Evaluate on Emotion Dataset 3
python examples/evaluate_models_dataset3.py

# Compare original vs. grayscale
python examples/compare_original_vs_greyscale.py
```

### 5. Interactive Dashboard

```bash
streamlit run src/visualization/app.py
```

## Using the Trained Model

### Download Weights

The trained Custom-ResNet18 checkpoint is included in the repository at `runs/exp2_clean/best.pt` (43 MB).

[TODO: If weights are also hosted externally (Hugging Face / GitHub Releases), add a download link here.]

### Inference Example

```python
from src.emotion_detection import EmotionDetector

detector = EmotionDetector()

# Single image
result = detector.detect_emotion("path/to/face.jpg")
print(result["dominant_emotion"])

# Batch processing
results = detector.detect_emotions_batch("path/to/image_folder/")
detector.save_results(results, "predictions.csv")
```

## Project Structure

```
.
├── src/
│   ├── emotion_detection/
│   │   └── detector.py            # EmotionDetector class (inference API)
│   ├── data_processing/           # Data quality assessment scripts
│   └── visualization/
│       └── app.py                 # Streamlit dashboard
├── models/
│   └── private_modle.py           # ResNet18 training script
├── experiments/
│   ├── configs/                   # 12 YAML experiment configurations
│   ├── runs/                      # Experiment outputs (weights, metrics)
│   ├── results_figures/           # Generated plots and tables
│   ├── run_experiment.py          # Single experiment runner
│   ├── run_batch.py               # Batch experiment runner
│   └── visualize_*.py             # Visualization scripts
├── runs/
│   └── exp2_clean/                # Best trained model checkpoint
├── results/                       # Evaluation reports (ensemble, per-dataset)
├── examples/                      # Example and evaluation scripts
├── docs/                          # Additional documentation and images
├── data/                          # Datasets (not tracked in git)
│   ├── raw/
│   └── processed/
├── requirements.txt
└── .gitignore
```

## Limitations and Lessons Learned

- **Class imbalance**: `disgusted` has very few samples (2.3% of Dataset 3), and all models perform worst on this class. Focal loss and weighted sampling provided marginal improvements.
- **Ensemble overhead vs. gain**: The override ensemble adds complexity but yields only +0.25% accuracy on the Faces Dataset. A learned combination or stacking approach could be more effective.
- **Dataset bias**: Models were trained on aligned, cropped faces. Performance degrades on in-the-wild images where face detection must first succeed.
- **Grayscale vs. color**: Comparison experiments are included; pretrained models (DeepFace, FER) were designed for specific input formats, limiting direct comparison.

### Possible Next Steps

- Train on larger, more balanced datasets (e.g., AffectNet, RAF-DB)
- Experiment with larger architectures (ResNet50, EfficientNet)
- Replace the rule-based ensemble with a learned stacking approach
- Add real-time webcam inference

## Acknowledgements

Developed as part of the **Data Science Project** course at **HSLU** ([TODO: semester]).

## Open TODOs

The following items require manual completion before the repository is fully portfolio-ready:

- [ ] Add GitHub handles for all team members in the Team section
- [ ] Add semester/term to Acknowledgements
- [ ] Add source links for Faces Dataset and Emotion Dataset 3
- [ ] Add external hosting link for model weights (if applicable)
- [ ] Optionally add individual contribution descriptions per team member

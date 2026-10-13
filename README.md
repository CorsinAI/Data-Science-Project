# Facial Emotion Recognition Pipeline

End-to-end system for detecting emotions in facial images using pretrained and custom-trained deep learning models, with an interactive dashboard and ensemble inference.

## Background

Recognizing human emotions from facial expressions is a long-standing challenge in computer vision. Early benchmark datasets like FER2013 established the 7-class taxonomy (angry, disgusted, fearful, happy, neutral, sad, surprised) that most research still follows, while larger-scale collections such as AffectNet and RAF-DB pushed state-of-the-art accuracies into the low-to-mid 60s on in-the-wild images. On the tooling side, open-source libraries -- notably [DeepFace](https://github.com/serengil/deepface) and [FER](https://github.com/justinshenk/fer) -- have made pretrained emotion classifiers accessible to anyone with a `pip install`.

Despite this progress, practical accuracy on naturalistic faces remains modest: even well-tuned CNNs rarely exceed 70% across all seven classes, and minority emotions like *disgusted* are consistently underrecognized. This project explores that gap. We compare two off-the-shelf detectors with a custom-trained ResNet18, combine them in a simple ensemble, and document the trial-and-error involved in hyperparameter tuning, class-imbalance mitigation, and model agreement analysis.

## Team and Acknowledgements

Developed at **HSLU** as part of the **Data Science Project** course ([TODO: semester]).

Members: Marco Blum, Kim Bortis, Lukas Pfruender, Corsin Hidber

## Getting Started

### 1. Clone and Install

```bash
git clone <repository-url>
cd <repository-name>
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note**: Python 3.12 is recommended. The requirements file pins PyTorch with CUDA 12.4. If your system uses a different CUDA version or CPU only, install PyTorch separately following [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

### 2. Run the Dashboard

```bash
streamlit run src/visualization/app.py
```

That's it -- the dashboard should open in your browser.

### 3. Data Download

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

| Dataset | Size | Format | Notes |
|---------|------|--------|-------|
| Faces Dataset (primary) | 35,887 images (28,709 train / 7,178 test) | PNG, 7 emotion folders per split | [TODO: source link] |
| Emotion Dataset 3 | 15,339 images (12,271 train / 3,068 test) | 48x48 aligned JPEG + CSV labels | `happy` 38.8%, `disgusted` 2.3%. [TODO: source link] |
| Faces Dataset 2 (supplementary) | 172 high-quality images | Includes demographic metadata (gender, age, country) | Quality benchmarking and bias analysis |

### 4. Download Weights

The trained Custom-ResNet18 checkpoint is included in the repository at `runs/exp2_clean/best.pt` (43 MB).

[TODO: If weights are also hosted externally (Hugging Face / GitHub Releases), add a download link here.]

### 5. Quick Inference Example (Optional)

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

## The Streamlit Dashboard

The interactive dashboard (`src/visualization/app.py`) provides four modes:

- **Browse Dataset** -- Navigate training and test images by emotion class. Useful for visually inspecting data quality and class distribution before any modeling.
- **Batch Analysis** -- Run all three models (DeepFace, FER, Custom-ResNet18) across an entire dataset split and view per-image predictions, agreement statistics, and confusion matrices.
- **Upload Image** -- Drag-and-drop a single face image and get instant predictions from each model side by side. Good for quick sanity checks or demonstrations.
- **Video Analysis** -- Process a video file frame-by-frame, detect faces, and overlay emotion predictions in real time.

## Our Approach

We compared three detection strategies, ranging from fully pretrained to custom-trained:

**DeepFace-Emotion** is the emotion analysis backend from the [DeepFace](https://github.com/serengil/deepface) library. It ships a pretrained CNN and requires no training on our side. Out of the box it reached 53% accuracy on the Faces Dataset test set -- useful as a baseline but clearly limited.

**FER** (the [Facial Expression Recognition](https://github.com/justinshenk/fer) library) provides another pretrained CNN. It performed noticeably better at 63%, likely because its training data overlaps more closely with the kinds of images in our test set.

**Custom-ResNet18** is where we spent most of our effort. We took a ResNet18 backbone pretrained on ImageNet and replaced the final fully-connected layer with a custom classification head for 7 emotions. ResNet18 was chosen deliberately over larger architectures: it trains in reasonable time on a consumer GPU (8 GB VRAM) while still providing enough representational capacity for this task. We used AdamW with OneCycleLR scheduling, cross-entropy with label smoothing (0.1), dropout (0.3), weight decay (0.05), and MixUp augmentation. This model reached 67.59% accuracy, and our best checkpoint (`runs/exp2_clean/`) scored 68.05% with a macro F1 of 65.30%.

**Ensemble strategy.** Rather than averaging probabilities, we used a simple override rule: Custom-ResNet18 serves as the base predictor, and its prediction is overridden only when both DeepFace and FER agree on a different label. On the Faces Dataset test set this produced a net gain of +17 correct predictions out of 6,684, raising accuracy from 67.59% to 67.85%. Honest assessment: the +0.25% gain barely justifies the added complexity. A learned stacking approach would likely do better.

## Experiments and Lessons Learned

We ran a systematic hyperparameter search across 12 configurations, varying learning rate, batch size, epoch count, loss function, augmentation strategy, and class balancing method. Full details are in [`experiments/EXPERIMENTS_SUMMARY.md`](experiments/EXPERIMENTS_SUMMARY.md).

### Training Setup

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

### What We Tried and What We Learned

**Heavy augmentation helped the most.** The `aug_heavy` configuration (aggressive random crops, color jitter, horizontal flips) produced the best single result at 68.36% accuracy / 65.52% macro F1. This was consistently above the baseline across runs, suggesting the model was mildly overfitting with standard augmentation.

**Focal loss was a net positive for minority classes.** Switching from cross-entropy to focal loss (`loss_focal`) improved F1 on underrepresented emotions like *disgusted* and *fearful*, pushing overall macro F1 to 65.13%. However, it slightly hurt accuracy on majority classes, so the accuracy gain was modest (67.61% vs. 66.85% baseline).

**Class imbalance remained our biggest headache.** `disgusted` accounts for only 2.3% of Dataset 3. We tried inverse-log class weighting and weighted random sampling (`balance_weighted_sampler`); both helped recall on rare classes but introduced instability in training loss. No single strategy fully resolved the imbalance.

**Longer training showed diminishing returns.** Extending from 30 to 50 epochs (`epochs_50_extended`) gave 67.35% -- barely above baseline -- and validation loss began creeping up after epoch 35. The OneCycleLR schedule appears well-matched to 30 epochs.

**Smaller batches were slightly better.** Batch size 64 (`batch_64_smaller`) reached 67.46% / 65.00% F1, outperforming batch size 256 (`batch_256_larger`). The smaller batch likely provides a regularization effect through noisier gradient estimates.

**The ensemble gain was marginal.** Adding DeepFace and FER to the override ensemble changed only 17 predictions on a 6,684-image test set. The models tend to agree on easy images and disagree on genuinely ambiguous ones, limiting the scope for ensemble correction.

### Reproducing the Experiments

```bash
# Single experiment (baseline)
python experiments/run_experiment.py experiments/configs/baseline.yaml \
    --data_root data/raw/Faces_Dataset

# All 12 experiments
python experiments/run_batch.py --data_root data/raw/Faces_Dataset
```

Outputs (weights, metrics, confusion matrices) are saved under `experiments/runs/<experiment_name>/`.

### Evaluation Scripts

```bash
# Evaluate all three models on Faces Dataset
python examples/ensemble_faces_dataset.py

# Evaluate on Emotion Dataset 3
python examples/evaluate_models_dataset3.py

# Compare original vs. grayscale
python examples/compare_original_vs_greyscale.py
```

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

All metrics are taken directly from experiment logs and evaluation reports in the repository.

### Plots

![Model Comparison](docs/images/overall_comparison.png)

![Per-Emotion F1 Heatmap](docs/images/test_f1_heatmap.png)

![Model Agreement by Emotion](docs/images/agreement_by_emotion.png)

## Limitations

- **Class imbalance**: `disgusted` has very few samples (2.3% of Dataset 3), and all models perform worst on this class. Focal loss and weighted sampling provided marginal improvements.
- **Ensemble overhead vs. gain**: The override ensemble adds complexity but yields only +0.25% accuracy on the Faces Dataset. A learned combination or stacking approach could be more effective.
- **Dataset bias**: Models were trained on aligned, cropped faces. Performance degrades on in-the-wild images where face detection must first succeed.
- **Grayscale vs. color**: Comparison experiments are included; pretrained models (DeepFace, FER) were designed for specific input formats, limiting direct comparison.

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

## Open TODOs

The following items require manual completion before the repository is fully portfolio-ready:

- [ ] Add GitHub handles for all team members in the Team section
- [ ] Add semester/term to Acknowledgements
- [ ] Add source links for Faces Dataset and Emotion Dataset 3
- [ ] Add external hosting link for model weights (if applicable)
- [ ] Optionally add individual contribution descriptions per team member

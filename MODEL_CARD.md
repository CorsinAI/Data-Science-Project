# Model Card: Custom-ResNet18 Emotion Classifier

## Model Details

- **Architecture**: ResNet18 (torchvision, ImageNet-pretrained) with a custom classification head (dropout 0.3 + linear layer)
- **Task**: 7-class facial emotion classification
- **Classes**: angry, disgusted, fearful, happy, neutral, sad, surprised
- **Input**: 224x224 RGB images
- **Framework**: PyTorch 2.6.0
- **Training script**: [`models/private_modle.py`](models/private_modle.py)
- **Checkpoint**: [`runs/exp2_clean/best.pt`](runs/exp2_clean/best.pt) (~43 MB)

## Training Data

- **Primary dataset**: Faces Dataset -- 28,709 training images across 7 emotion classes
- **Format**: PNG images organized into class folders
- **Preprocessing**: Resize to 224x224, normalize with ImageNet mean/std
- **Augmentation**: Random horizontal flip, random crop, MixUp (alpha=0.2)

## Training Procedure

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| LR schedule | OneCycleLR (pct_start=0.3) |
| Batch size | 128 |
| Epochs | 30 |
| Weight decay | 0.05 |
| Label smoothing | 0.1 |
| Loss | CrossEntropyLoss with class weights |
| Early stopping | Patience 7 (on validation accuracy) |
| Seed | 42 |
| Hardware | Local GPU, 8 GB VRAM |

## Evaluation Results

### Best Checkpoint (runs/exp2_clean/)

| Metric | Value |
|--------|-------|
| Test accuracy | 68.05% |
| Test macro F1 | 65.30% |

### Comparison with Pretrained Models (Faces Dataset test set)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| DeepFace-Emotion | 53.13% | 48.76% |
| FER | 63.33% | 58.95% |
| Custom-ResNet18 | 67.59% | 63.97% |

## Intended Use

- Academic evaluation and portfolio demonstration
- Single-image and batch emotion prediction on cropped, aligned face images
- Baseline for further research on facial emotion recognition

## Limitations

- Trained on a single dataset; may not generalize well to other face distributions
- Performance is lowest on `disgusted` (severely underrepresented in training data at ~2.3%)
- Expects pre-cropped, frontal face images. Performance degrades on uncropped or profile images.
- Not validated for real-time or production use

## Ethical Considerations

- Emotion recognition from faces is an inherently uncertain task. Predicted labels should not be treated as ground truth about a person's internal state.
- The training data may contain demographic biases that affect per-group performance.
- This model is intended for academic purposes only and should not be used for surveillance, hiring decisions, or other high-stakes applications.

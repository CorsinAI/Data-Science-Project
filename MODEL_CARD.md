# Model Card: Custom-ResNet18 Emotion Classifier

## Model Details

- **Architecture**: ResNet18 (torchvision, ImageNet-pretrained) with a custom classification head (dropout 0.3 + linear layer)
- **Task**: 7-class facial emotion classification (angry, disgusted, fearful, happy, neutral, sad, surprised)
- **Input**: 224x224 RGB images
- **Framework**: PyTorch 2.6.0
- **Training script**: [`models/private_model.py`](models/private_model.py)
- **Checkpoint**: `runs/exp2_clean/best.pt` (~43 MB, not tracked in git)

For training setup, hyperparameter details, and full results see the [README](README.md).

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

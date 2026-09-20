# AI-Based Image Classification System Using Deep Learning

## Project Objective
Build a deep learning-based image classification system using Convolutional Neural Networks (CNN) and Transfer Learning, and additionally demonstrate an LSTM-based sentiment analysis model and a Hugging Face pretrained pipeline. The project covers preprocessing, model building, training, evaluation, and prediction visualization.

## Dataset Description
- **Name:** CIFAR-10
- **Source:** https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz (loaded in-notebook via `tensorflow.keras.datasets.cifar10`, which serves the same official dataset)
- **Size:** 60,000 32x32 color images across 10 classes (50,000 train / 10,000 test)
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Additional dataset (for LSTM task):** IMDB Movie Reviews dataset, loaded via `tensorflow.keras.datasets.imdb` — 25,000 train / 25,000 test labeled reviews (positive/negative).

## Methodology
1. **Data Preparation:** Loaded CIFAR-10, explored class distribution, normalized pixel values (0–1), split into train/validation/test sets, visualized sample images.
2. **CNN Model:** Built a custom CNN (Conv2D → MaxPooling → Dropout, repeated, then Flatten → Dense → Softmax), trained from scratch, evaluated with accuracy, loss, confusion matrix, and classification report.
3. **Transfer Learning:** Loaded MobileNetV2 pretrained on ImageNet, froze its base layers, added a custom classification head, resized CIFAR-10 images to 96x96, trained and evaluated the same way, and compared results against the custom CNN.
4. **Prediction & Visualization:** Predicted on unseen test images, displayed actual vs predicted labels, visualized correctly classified and misclassified images, and plotted training/validation accuracy and loss curves for both models.
5. **LSTM Sentiment Analysis:** Built an Embedding + LSTM model from scratch on the IMDB dataset for binary sentiment classification.
6. **Hugging Face Pipeline:** Used the pretrained `sentiment-analysis` pipeline (`transformers` library) on sample reviews, without any training, for comparison against the from-scratch LSTM model.

## Model Architecture
**Custom CNN:**
```
Conv2D(32) -> MaxPool -> Dropout
Conv2D(64) -> MaxPool -> Dropout
Conv2D(128) -> MaxPool -> Dropout
Flatten -> Dense(256) -> Dropout -> Dense(10, softmax)
```

**Transfer Learning:**
```
MobileNetV2 (frozen, pretrained on ImageNet)
-> GlobalAveragePooling2D
-> Dense(128) -> Dropout -> Dense(10, softmax)
```

**LSTM (Sentiment):**
```
Embedding(10000, 64) -> LSTM(64) -> Dense(1, sigmoid)
```

## Results and Analysis
- Both the custom CNN and the MobileNetV2 transfer-learning model were evaluated on accuracy, loss, confusion matrix, and classification report (exact numbers depend on the run — see notebook outputs after execution).
- Transfer Learning generally converges faster and can reach reasonable accuracy in fewer epochs since it reuses ImageNet-pretrained features, at the cost of extra preprocessing (resizing small 32x32 CIFAR images to 96x96).
- The custom CNN has full architectural flexibility but typically needs more epochs/data to match transfer learning's early performance.
- The Hugging Face pretrained pipeline gave instant sentiment predictions with no training required, while the LSTM model needed to be trained from scratch on the IMDB dataset — illustrating the trade-off between using pretrained models vs. training custom models.

## Conclusion
This project demonstrates the two core deep learning approaches — training a model from scratch (CNN, LSTM) and reusing pretrained models (MobileNetV2, Hugging Face pipeline) — across both image and text classification tasks, along with proper preprocessing, evaluation, and result visualization.

## How to Run
1. Open `AI_Image_Classification_Project.ipynb` in Jupyter Notebook, Google Colab, or Kaggle.
2. Run all cells in order (Runtime → Run all in Colab).
3. Training may take a few minutes depending on hardware; using a GPU runtime is recommended (Colab: Runtime → Change runtime type → GPU).

## Repository Contents
- `AI_Image_Classification_Project.ipynb` — full notebook with code and outputs
- `README.md` — this documentation file

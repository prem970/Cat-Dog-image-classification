# Image Classification: Cats vs. Dogs

This notebook demonstrates how to build and train a Convolutional Neural Network (CNN) to classify images of cats and dogs using TensorFlow/Keras.

## 1. Problem Overview
- **Task**: Binary image classification (distinguishing between images of cats and dogs).
- **Dataset**: Kaggle Cats and Dogs dataset containing thousands of images of various sizes and formats.
- **Challenge**: The raw dataset contains some corrupted, empty, or incorrectly formatted non-JFIF images that need to be programmatically cleaned before training to prevent crashes.

## 2. Approach
- **Data Prep & Cleaning**: Downloaded the dataset, scanned files, and removed corrupt images (1,590 images removed). Splitting the remaining data into an 80/20 train/validation split using `image_dataset_from_directory` with images resized to `150x150` pixels.
- **Model Architecture**: A 3-layer Convolutional Neural Network (CNN):
  - **Input Layer**: `150x150x3` with pixel rescaling (`1./255`).
  - **Feature Extraction**: Three successive blocks of `Conv2D` layers (with `relu` activation) followed by `MaxPooling2D` to extract hierarchical spatial features.
  - **Classification Head**: `Flatten` layer, a fully connected `Dense` layer with 128 units, a `Dropout` layer (rate `0.5`) to reduce overfitting, and a final single-unit `Dense` layer with `sigmoid` activation for binary classification.
- **Compilation**: Compiled using the `adam` optimizer, `binary_crossentropy` loss, and tracking `accuracy`.

## 3. Results
The model was trained for 5 epochs. Here is a summary of the performance:

- **Training Accuracy**: Climbed from **~61.0%** in Epoch 1 to **~86.5%** in Epoch 5.
- **Validation Accuracy**: Reached a stable peak of **~83.0%** by Epoch 5.
- **Loss Reduction**: Training loss dropped from `0.6547` to `0.3140`, while validation loss dropped from `0.5682` to `0.3915`.

*Note: The learning curves show strong, consistent convergence, although starting to show early signs of slight variance between train and validation metrics towards epoch 5, which is handled well by the dropout layer.*

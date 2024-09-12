# Cats vs Dogs Image Classification Project

This project aims to classify images of cats and dogs using a custom-built Convolutional Neural Network (CNN) model. The dataset consists of images of cats and dogs, and the objective is to create a model that can accurately distinguish between the two classes.

## Dataset

The dataset used for this project contains images of cats and dogs. Images were preprocessed and stored in a NumPy array with pixel values between 0 and 255.

## Requirements

The following libraries are needed for this project:

- TensorFlow
- NumPy
- Matplotlib
- Google Colab (optional)
- os (for directory management)

You can install the dependencies with:
```bash
pip install tensorflow numpy matplotlib
```

## Model Architecture

The CNN model used in this project is defined as follows:

### **Model Layers**:
1. **Conv2D Layer (32 filters, 3x3 kernel)**: Extracts features from the images.
2. **Batch Normalization**: Normalizes the activations and gradients to improve training.
3. **MaxPooling2D (2x2 pool size)**: Reduces the spatial dimensions of the feature maps.
4. **Conv2D Layer (64 filters, 3x3 kernel)**: Further feature extraction with increased filter depth.
5. **Batch Normalization**: Helps stabilize the learning process.
6. **MaxPooling2D (2x2 pool size)**: Reduces dimensions.
7. **Conv2D Layer (128 filters, 3x3 kernel)**: Deep feature extraction.
8. **Batch Normalization**: Continues normalization to improve training efficiency.
9. **MaxPooling2D (2x2 pool size)**: Further reduces spatial dimensions.
10. **Flatten**: Converts the 3D feature maps into a 1D feature vector.
11. **Dense (128 units)**: Fully connected layer with 128 units.
12. **Dropout (0.5)**: Dropout regularization to prevent overfitting.
13. **Dense (64 units)**: Another fully connected layer with 64 units.
14. **Dropout (0.5)**: Dropout for additional regularization.
15. **Output Layer (1 unit)**: Binary classification using a sigmoid activation function.

### Model Summary:
- **Total Parameters**: 14,848,193
- **Trainable Parameters**: 14,847,745
- **Non-trainable Parameters**: 448

### Optimizer and Loss Function:
- Optimizer: Adam
- Loss Function: Binary Crossentropy

## Data Preprocessing

1. **Image Generators**: ImageDataGenerators were used to load and preprocess the dataset, allowing for real-time data augmentation and efficient memory usage.
   
2. **Normalization**: The images were normalized by dividing the pixel values by 255 to rescale the range to [0, 1].
   
   ```python
   def process(image, label):
       image = tf.cast(image / 255.0, tf.float32)
       return image, label

   train_ds = train_ds.map(process)
   validation_ds = val_ds.map(process)
   ```

3. **Augmentation**: Basic augmentations such as random flips and rotations were applied to make the model more robust to variations in the input images.

## Training and Validation

- The dataset was split into training and validation sets.
- The model was trained using a batch size that can be adjusted in the code.
- Real-time image augmentation was applied during training to help the model generalize better to unseen images.

## Evaluation

- After training, the model was evaluated on a test set, and performance metrics like accuracy and loss were tracked.
- The results were visualized using **Matplotlib**, showing training history (accuracy and loss curves).

## Results

- The final model performance on the test set is provided through accuracy and loss plots.
- Confusion matrices and classification reports were generated for more detailed insights into the model's predictions.

## How to Run

1. Open the notebook in Google Colab or a local Jupyter environment.
2. Connect your Google Drive to access the dataset using:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Make sure the dataset is properly organized and accessible.
4. Run the notebook cells sequentially to load the dataset, preprocess the images, and train the model.
5. Use Matplotlib to visualize the results, including accuracy and loss metrics over the training process.

## Contributors

- Mohtasham Murshid Madani

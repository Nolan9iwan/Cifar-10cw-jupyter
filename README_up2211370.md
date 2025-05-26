# CIFAR-10 Image Classification with CNN

# Introduction
This project implements an end-to-end machine learning pipeline for image classification using a Convolutional Neural Network (CNN). The dataset used is CIFAR-10, which contains 60,000 32x32 colour images in 10 classes, with 6,000 images per class. The goal is to train a CNN that can accurately classify unseen images.

# Business Objectives
The objective is to build a working image classifier that can distinguish between 10 object categories (e.g., cat, airplane, truck). The goal is to achieve high classification accuracy and demonstrate model generalisation to unseen test data.


# ML Pipeline

# 1. Data Collection and Preparation
- Used `tensorflow.keras.datasets` to load CIFAR-10 dataset.
- Normalized pixel values from `[0, 255]` to `[0, 1]`.
- Split data into training and test sets.

# 2. Exploratory Data Analysis (EDA)
- Displayed 9 sample images with class labels to inspect data visually.
- Plotted the number of images per class using a bar chart to confirm class balance.

# 3. Model Building
- Constructed a CNN using Keras Sequential API.
- Architecture included:
  - 2 Conv2D + MaxPooling2D layers
  - Flatten layer
  - Dense layer with Dropout
  - Output layer with softmax activation
- Compiled using `Adam` optimizer and `sparse_categorical_crossentropy`.

# 4. Model Evaluation
- Trained the model over 10 epochs.
- Plotted training and validation accuracy and loss over epochs.
- Evaluated final performance on test set using `.evaluate()` method.
- Printed confusion matrix and classification report to assess per-class performance.

# 5. Prediction
- Used trained model to predict on unseen test images.
- Displayed 5 sample images with predicted and actual labels.

## Jupyter Notebook Structure

The notebook is organised into five sections matching the coursework requirements:

1. Data Collection and Preparation: Loads and normalises the CIFAR-10 dataset.
2. Exploratory Data Analysis (EDA): Displays sample images and visualises class distribution.
3. Model Building: Constructs the CNN using Keras Sequential API.
4. Model Evaluation: Trains the model, plots accuracy/loss, and evaluates performance.
5. Prediction: Tests the model on unseen test data and shows example predictions.

Markdown headings were used to separate each section clearly.


## Future Work

If more time were available, I would:
- Try deeper CNN architectures (more layers or filters)
- Use techniques like early stopping and learning rate scheduling

# Libraries and Modules

- tensorFlow / Keras  
  Used to build, train, and evaluate the CNN model. `keras.Sequential()` was used to add layers.  

- NumPy
  Used for numerical operations such as reshaping, normalisation, and applying `argmax` to extract class labels from predictions.

- Matplotlib  
  Used to visualise sample images and model predictions, and to plot graphs of training loss and accuracy.

- Seaborn  
  Used to create better-styled plots for class distribution and training metrics.

- Pandas  
  Used to turn the training history dictionary into a DataFrame so that training accuracy/loss could be visualised easily.

- Scikit-learn (sklearn)  
  Used for calculating the confusion matrix and generating the classification report to evaluate per-class performance.


# Unfixed Bugs

- No major bugs encountered. All code ran successfully in Google Colab.  

# Acknowledgements and References

- Some structure and code guidance was based on examples provided in lectures and coursework support materials.
- Code for plotting loss and accuracy was adapted from course templates.
- [ChatGPT](https://chat.openai.com/) was used for occasional explanation and formatting support.
- CIFAR-10 dataset is provided via `tensorflow.keras.datasets`.

# Conclusions

The project successfully demonstrates an end-to-end machine learning pipeline using a CNN to classify CIFAR-10 images. The model achieved good accuracy on unseen test data and generalised well. Each required step (from data collection to prediction) was completed and documented in the notebook. If more time were available, deeper model tuning and advanced features could further improve performance.

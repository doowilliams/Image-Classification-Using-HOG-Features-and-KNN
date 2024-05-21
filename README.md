# Image-Classification-Using-HOG-Features-and-KNN

This project implements a binary image classifier to distinguish between images of cats and dogs. It uses Histogram of Oriented Gradients (HOG) features and a k-nearest neighbors (KNN) classifier, incorporating k-means clustering and PCA for feature extraction and visualization.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Critique and Recommendation](#critique-and-recommendation)
- [Acknowledgements](#acknowledgements)

## Installation

To run this project, you need to have Python installed along with several libraries. You can install the required libraries using pip:

```bash
pip install numpy opencv-python scikit-learn matplotlib scikit-image
```

## Usage

1. **Unzip Dataset**:
   Unzip the dataset before loading images.

   ```python
   import zipfile

   zip_ref = zipfile.ZipFile("/content/drive/MyDrive/dataset/cat_dog.zip", "r")
   zip_ref.extractall()
   zip_ref.close()
   ```

2. **Load Images**:
   Load and preprocess images from the dataset.

   ```python
   import os
   import cv2
   import numpy as np

   def load_images(path, limit=None):
       images = []
       classes = []
       count = 0
       for filename in os.listdir(path):
           if limit is not None and count >= limit:
               break
           if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
               category = filename.split('.')[0]
               cls = 1 if category.lower() == 'dog' else 0
               img_path = os.path.join(path, filename)
               image = cv2.imread(img_path)
               if image is not None:
                   image = cv2.resize(image, (128, 128))
                   images.append(image)
                   classes.append(cls)
                   count += 1
       return np.array(images), np.array(classes)

   images, classes = load_images("/content/cat_dog")
   ```

3. **Train-Test Split**:
   Split the dataset into training and testing sets.

   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(images, classes, train_size=0.8, test_size=0.2, random_state=42)
   ```

4. **Extract HOG Features**:
   Extract HOG features from images.

   ```python
   from skimage.feature import hog

   def hog_features(image):
       features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
       return features

   hog_features_train = [hog_features(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) for image in X_train]
   ```

5. **K-means Clustering and PCA**:
   Perform k-means clustering and PCA on HOG features.

   ```python
   from sklearn.cluster import MiniBatchKMeans
   from sklearn.decomposition import PCA

   kmeans = MiniBatchKMeans(n_clusters=100, random_state=42)
   kmeans.fit(hog_features_train)
   pca = PCA(n_components=2)
   hog_features_pca = pca.fit_transform(hog_features_train)
   ```

6. **Bag of Visual Words (BoVW) Representation**:
   Convert images into BoVW representation.

   ```python
   def bovw_representation(images, kmeans, n_clusters=100):
       histograms = []
       for image in images:
           gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
           features = hog_features(gray_image)
           histogram, _ = np.histogram(kmeans.predict([features]), bins=range(n_clusters+1))
           histograms.append(histogram)
       return np.array(histograms)

   x_train_bovw = bovw_representation(X_train, kmeans)
   x_test_bovw = bovw_representation(X_test, kmeans)
   ```

7. **Train KNN Classifier**:
   Train a KNN classifier using BoVW features.

   ```python
   from sklearn.neighbors import KNeighborsClassifier

   KNN_classifier = KNeighborsClassifier(n_neighbors=40)
   KNN_classifier.fit(x_train_bovw, y_train)
   ```

8. **Make Predictions and Evaluate**:
   Evaluate the classifier on the test set.

   ```python
   from sklearn.metrics import accuracy_score, classification_report

   y_pred = KNN_classifier.predict(x_test_bovw)
   accuracy = accuracy_score(y_test, y_pred)
   print("Accuracy:", accuracy)
   print(classification_report(y_test, y_pred))
   ```

9. **Combined Histogram of Visual Words**:
   Generate and plot a combined histogram of visual words.

   ```python
   def bag_of_visual_words(images, n_clusters=100):
       sift = cv2.SIFT_create()
       descriptors = []
       for img in images:
           kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
           if des is not None:
               descriptors.append(des)
       descriptors = np.vstack(descriptors)
       kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=100, max_iter=100)
       kmeans.fit(descriptors)
       histograms = []
       for des in descriptors:
           histogram, _ = np.histogram(kmeans.predict(des.reshape(1, -1)), bins=range(n_clusters+1))
           histograms.append(histogram)
       histograms = np.vstack(histograms)
       combined_histogram = np.sum(histograms, axis=0)
       normalized_histogram = combined_histogram / np.sum(combined_histogram)
       return normalized_histogram

   histograms = bag_of_visual_words(images)
   plt.figure(figsize=(10, 6))
   plt.bar(range(len(histograms)), histograms)
   plt.xlabel('Visual Word Index')
   plt.ylabel('Frequency')
   plt.title('Combined Histogram of Visual Words')
   plt.show()
   ```

## Project Structure

```
├── dataset
│   ├── cat_dog.zip
├── main.py
├── README.md
└── requirements.txt
```

## Methodology

1. **Data Preparation**: Unzip and load the dataset, resizing images to 128x128 pixels.
2. **Feature Extraction**: Use HOG to extract features from grayscale images.
3. **Clustering**: Apply k-means clustering to the HOG features to create a visual vocabulary.
4. **Dimensionality Reduction**: Use PCA for visualization of the HOG features.
5. **BoVW Representation**: Convert images into histograms of visual words.
6. **Classification**: Train a KNN classifier on the BoVW histograms.
7. **Evaluation**: Evaluate the classifier performance on the test set.
8. **Visualization**: Generate a combined histogram of visual words for all images.

## Results

The KNN classifier achieved an accuracy of approximately `X%` on the test set. Detailed classification metrics are provided in the classification report.

## Critique and Recommendation

### Critique

- **Feature Selection**: HOG features are robust but may not handle all variations in imaging conditions, such as scale and rotation.
- **Classifier**: KNN is simple but memory-intensive and may not be suitable for large datasets.
- **Computational Efficiency**: The feature extraction and clustering processes are computationally intensive.
- **Interpretability**: The model lacks interpretability, making it difficult for non-technical stakeholders to understand its decisions.

### Recommendation

1. **Feature Selection**: Consider using convolutional neural networks (CNNs) for feature extraction to handle more variations.
2. **Classifier**: Explore more sophisticated classifiers like support vector machines (SVM) or deep neural networks (DNN) for better performance.
3. **Computational Efficiency**: Optimize the clustering algorithm or use a smaller vocabulary size to reduce computational overhead.
4. **Involvement of Moderators**: Develop features to involve online content moderators in the system.
5. **Interpretability**: Incorporate techniques for model interpretability, such as attention mechanisms or feature visualization.

## Acknowledgements

- Petfinder.com for providing the dataset.
- Open-source libraries: OpenCV, scikit-learn, matplotlib, and scikit-image.

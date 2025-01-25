# Generating Machine Learning Models Using Machine Learning Models

## Overview
This project explores the innovative concept of leveraging CycleGANs to merge the feature representations of two Convolutional Neural Networks (CNNs) trained on distinct domains. The resulting model is capable of performing a novel task—in this case, detecting black cats—by combining knowledge from CNNs trained to recognize cats and the color black.

The project introduces a new perspective on generative model applications for machine learning, opening pathways for automating the creation of task-specific classifiers while tackling challenges like data scarcity and limited computational resources.

---

## Key Features
- **CycleGANs for Model Fusion**: Utilizes CycleGANs to transfer knowledge between CNNs trained on different domains.
- **Generated CNNs**: Creates task-specific models by combining learned features from existing models.
- **Dimensionality Reduction and Clustering**: Applies UMAP, K-means, and DBSCAN to analyze and validate feature spaces.
- **Unsupervised Learning**: Demonstrates the ability to recognize black cats without direct labeled training data.

---

## Methodology
1. **Datasets**:
   - Black/Random Images: 1,826 samples (1,745 black, 81 random).
   - Cat/Random Images: 30,405 samples (29,843 cats, 562 random).
   - Kernels Extracted for CycleGAN Training: 4,498 sets for each convolutional layer.

2. **Model Architectures**:
   - **CNNs**: Two convolutional layers with 5x5 kernels, ReLU activation, and max-pooling.
   - **CycleGANs**: Generators and discriminators designed for feature domain translation.
   - **Generated CNN**: Utilizes CycleGAN-generated kernels for feature extraction.

3. **Evaluation Metrics**:
   - Cluster Entropy and Purity
   - Accuracy, Precision, Recall

---

## Results
- The generated CNN successfully clustered black cat images, showcasing effective unsupervised learning.
- Cosine similarity and UMAP visualizations validated the interpretability of the learned feature space.

---

## Future Work
- Optimize hyperparameters for clustering and feature extraction.
- Explore alternative metrics for feature space analysis.
- Develop a semantic-based end-to-end pipeline.

---

## Setup and Execution
### Requirements
- **Hardware**: Minimum GPU with 6GB VRAM recommended.
- **Software**:
  - Python 3.11
  - PyTorch 2.5, torchvision
  - Numpy, scikit-learn, seaborn, matplotlib, tqdm

---

## Authors
- Aditya Patel
- Karan Jain

This project was completed as part of **CS 271: Topics in Machine Learning** under the guidance of **Prof. Mark Stamp** at **San Jose State University**.


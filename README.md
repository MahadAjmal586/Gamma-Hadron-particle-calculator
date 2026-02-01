<div align="center">

# Gamma vs Hadron Classification (MAGIC Telescope Data)

![last-commit](https://img.shields.io/github/last-commit/MahadAjmal586/Gamma-Hadron-Classification?style=flat&logo=git&logoColor=white&color=00c853)
![repo-top-language](https://img.shields.io/github/languages/top/MahadAjmal586/Gamma-Hadron-Classification?style=flat&color=00c853)
![repo-language-count](https://img.shields.io/github/languages/count/MahadAjmal586/Gamma-Hadron-Classification?style=flat&color=00c853)

**Built with:**

![Python](https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458.svg?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C.svg?style=flat)
![Imbalanced--Learn](https://img.shields.io/badge/Imbalanced--Learn-FF6F00.svg?style=flat)

<br>

A machine learning project that classifies **gamma-ray events vs hadronic background** using **K-Nearest Neighbors** and **Naive Bayes**, with visualization, feature scaling, and class balancing.

</div>

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Data Visualization](#data-visualization)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Models Used](#models-used)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)

---

## Project Description

This project focuses on **binary classification of astronomical events** recorded by the MAGIC Gamma Telescope.  
The goal is to distinguish **gamma-ray signals** from **hadronic background noise** using machine learning models.

The workflow includes:
- Exploratory data analysis
- Feature-wise distribution visualization
- Feature scaling
- Class imbalance handling using oversampling
- Model training and evaluation

This is a great example of a **real-world ML classification pipeline**.

---

## Dataset

- File: `telescope_data.csv`
- Target column: `class`
  - `g` → Gamma ray (1)
  - `h` → Hadron (0)

### Preprocessing
- Converted target labels to binary values
- Checked data types and statistics
- Verified class distribution
- Standardized all numerical features

---

## Tech Stack

- **Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib  
- **Machine Learning:** Scikit-learn  
- **Class Imbalance Handling:** Imbalanced-learn  

---

## Data Visualization

Feature-wise histograms are plotted for both classes:
- **Gamma events**
- **Hadron events**

This helps understand how each feature contributes to class separation.

Each plot shows:
- Probability density
- Overlapping class distributions
- Feature-level insight

---

## Preprocessing Pipeline

- **Standard Scaling:**  
  All features are scaled using `StandardScaler`

- **Oversampling:**  
  Applied **RandomOverSampler** only on the training set to handle class imbalance

- **Data Split:**
  - 60% Training
  - 20% Validation
  - 20% Testing

---

## Models Used

### 1. K-Nearest Neighbors (KNN)
- Number of neighbors: `k = 5`
- Distance-based classification
- Sensitive to feature scaling (handled)

### 2. Naive Bayes (GaussianNB)
- Probabilistic classifier
- Assumes feature independence
- Fast and effective baseline model

---

## Evaluation

Both models are evaluated using a **classification report**, including:
- Precision
- Recall
- F1-score
- Accuracy

This allows a fair comparison between:
- Distance-based learning (KNN)
- Probabilistic learning (Naive Bayes)

---

## Project Structure

```
Gamma-Hadron-Classification/
│
├── telescope_data.csv # Dataset
├── main.py / notebook.ipynb# Data analysis & models
├── README.md # Documentation
├── requirements.txt # Dependencies (recommended)
└── .gitignore
```

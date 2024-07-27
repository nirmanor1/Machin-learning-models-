### README

# Laptop Price and Condition Prediction using Machine Learning

Welcome to the Laptop Price and Condition Prediction repository! This project analyzes a dataset of laptops listed on eBay, aiming to predict the price and condition of laptops based on various features using machine learning techniques. 

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Machine Learning Algorithms](#machine-learning-algorithms)
- [Project Sections](#project-sections)
- [Features](#features)

## Introduction

This repository contains the code and resources for predicting laptop prices and conditions using machine learning models. The dataset includes information about laptop brands, specifications, conditions, and prices. The project involves data exploration, preprocessing, feature engineering, and applying various machine learning algorithms.

## Data

The data used in this project includes 2,939 entries with the following features:

- **Brand**: Manufacturer of the laptop
- **Product_Description**: Raw description extracted from eBay
- **Screen_Size**: Diagonal size of the laptop's display (in inches)
- **RAM**: Amount of Random Access Memory (in GB)
- **Processor**: Type and generation of the CPU
- **GPU**: Graphics processing unit
- **GPU_Type**: Indicates if the GPU is integrated or dedicated
- **Resolution**: Display resolution (width x height)
- **Condition**: Physical and operational state (e.g., New, Open box, Refurbished)
- **Price**: Cost of the laptop in USD

The data is split into training, validation, and test sets.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Project Structure

```plaintext
├── data
│   └── assignment-1-data.csv
├── notebooks
│   ├── DataExploration.ipynb
│   ├── DataPreprocessing.ipynb
│   ├── ModelTraining.ipynb
│   └── ModelEvaluation.ipynb
├── src
│   ├── data_exploration.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/laptop-price-prediction.git
cd laptop-price-prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Exploration**: Explore the dataset to understand its structure and contents. This step includes visualizing the data and identifying key patterns.
2. **Data Preprocessing**: Prepare the data for modeling by handling missing values, creating new features, and normalizing data.
3. **Feature Engineering**: Generate new features such as total pixels from the resolution.
4. **Model Training**: Train multiple machine learning models to predict laptop prices and conditions. Compare the performance of different models.
5. **Model Evaluation**: Evaluate the trained models using appropriate metrics and compare them with scikit-learn implementations.

## Machine Learning Algorithms

This project leverages various machine learning algorithms to analyze and predict laptop prices and conditions. Below are the algorithms used:

### Classification Algorithms

1. **Decision Tree Classifier**: A tree-like model used to predict the class of a target variable by learning simple decision rules inferred from data features.
2. **Random Forest Classifier**: An ensemble method that operates by constructing multiple decision trees and outputting the class that is the mode of the classes of the individual trees.

### Regression Algorithms

1. **Decision Tree Regressor**: Similar to the Decision Tree Classifier, but used for predicting continuous values.
2. **Random Forest Regressor**: An ensemble method that constructs multiple decision trees and averages their predictions for continuous target variables.

### Performance Evaluation

- **Accuracy**: Used for classification tasks to measure the number of correct predictions.
- **Mean Squared Error (MSE)**: Used for regression tasks to measure the average squared difference between predicted and actual values.
- **Sensitivity and Specificity**: Metrics for evaluating classification models, particularly in distinguishing between different classes.

## Project Sections

### Section A - Coding

**Decision Tree and Random Forest Implementation**

- Implemented both classifiers and regressors using recursive splitting based on Gini index and SSR (sum of squared residuals) respectively.
- Supported prediction by traversing the constructed trees.

### Section B - Data Visualization & Preparation

**Data Visualization**

- Explored data using visualizations to gain insights into the data.
- Created plots to visualize the distribution of brands, relationship between screen size and price, GPU type vs. price, and RAM vs. price.

**Data Preparation**

- Handled missing values using imputation.
- Converted resolution to total pixels for numerical representation.
- Encoded categorical features using label encoding.
- Applied KNN imputation to fill missing values in numerical features.

### Section C - Implementation

**Classification and Regression**

- Trained models to predict laptop condition and price using features such as Brand, Screen_Size, RAM, Processor, GPU, GPU_Type, Resolution, and Price.
- Conducted hyperparameter tuning for both classifiers and regressors to optimize model performance.
- Evaluated models using validation accuracy for classification and MSE for regression.

### Section D - Comparison

**Comparison with Scikit-Learn Models**

- Implemented Decision Tree and Random Forest models using scikit-learn.
- Compared custom implementations with scikit-learn models in terms of accuracy, MSE, and runtime.
- Analyzed differences in performance and runtime between custom and scikit-learn models.

### Section E - Bonus

**Screen Resolution and Metrics**

- Split the Resolution column into height and width.
- Implemented classification and regression tasks using the new features and compared results with initial models.
- Reported sensitivity and specificity metrics for classification models.
- Changed random forest regression to use median instead of mean and compared results.

## Features

- Comprehensive data exploration and visualization.
- Robust data preprocessing and feature engineering.
- Application of various machine learning models for prediction.
- Detailed comparison between custom implementations and scikit-learn models.
- In-depth analysis and discussion of results and insights.

# CAR-EVALUATION-PROJECT
This repository contains the code and documentation for a machine learning project aimed at evaluating cars using various classification methods. The project includes exploratory data analysis (EDA), preprocessing, model selection, and hyperparameter tuning.

## **Project Overview**

The goal of this project is to predict the car evaluation outcome based on several features using a classification approach. The dataset used is the Car Evaluation Dataset, which involves solving a multiclass classification problem.

## **Table of Contents**

- [Data Understanding](#data-understanding)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Selection](#model-selection)
- [Model Evaluation](#model-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Conclusion](#conclusion)

## **Data Understanding**

The dataset consists of categorical features related to car attributes. Key aspects include:

- **Features:** vhigh, vhigh.1, 2, 2.1, small, low
- **Target Variable:** unacc (with categories unacc, acc, good, vgood)

Exploratory Data Analysis (EDA) was conducted to visualize and understand the data distribution and relationships between features.

## **Data Preprocessing**

The preprocessing steps included:

- **Handling Missing Values:** No missing values were found.
- **Removing Duplicates:** Duplicates were removed to ensure data integrity.
- **Encoding Categorical Variables:** Used Label Encoding to convert categorical data into numerical values.
- **Feature Scaling:** Standardized features using StandardScaler.

## **Feature Engineering**

Key features were selected based on their importance and relevance to the target variable. Feature selection techniques included:

- **Correlation Analysis:** To identify and remove redundant features.
- **Importance Scores:** Evaluated feature importance using different models.

## **Model Selection**

Various classification models were evaluated, including:

- **Logistic Regression**
- **SVM (Support Vector Machine)**
- **Decision Tree**
- **Random Forest**
- **K-Nearest Neighbors**
- **Naive Bayes**
- **MLP (Multi-Layer Perceptron) Classifier**

### **Best Model**

The MLP Classifier yielded the highest accuracy compared to other models.

## **Model Evaluation**

The performance of each model was assessed using accuracy, precision, recall, and F1-score. The MLP Classifier achieved an accuracy of 99.42% before hyperparameter tuning.

### **Evaluation Metrics:**
- **Accuracy:** 99.42%
- **Precision, Recall, and F1-Score:** Detailed metrics provided in the classification report.

## **Hyperparameter Tuning**

Hyperparameter tuning was performed for the MLP Classifier using GridSearchCV. The tuning process involved:

- **Parameters Tuned:** hidden_layer_sizes, activation, solver, learning_rate, max_iter
- **Best Parameters:** Details of the optimal parameters and the resulting accuracy are provided.

## **Conclusion**

The project demonstrated the effectiveness of the MLP Classifier in achieving high accuracy. Hyperparameter tuning helped optimize the model performance further. The project provides a comprehensive approach to tackling classification problems with various machine learning techniques.

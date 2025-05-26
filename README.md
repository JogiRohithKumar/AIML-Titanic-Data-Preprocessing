# AIML-Titanic-Data-Preprocessing
Data cleaning and preprocessing of the Titanic dataset

### Titanic Dataset Preprocessing

This project is done as part of the *AI & ML Internship – Task 1: Data Cleaning and Preprocessing*.  
The goal is to clean and prepare the Titanic dataset for analysis or machine learning by handling missing values, encoding categorical variables, scaling features, and removing outliers.

---

## 📁 Dataset Description

The dataset is a CSV file (titanic.csv) containing details of passengers aboard the Titanic, including:

- PassengerId  
- Survived  
- Pclass  
- Name  
- Sex  
- Age  
- SibSp  
- Parch  
- Ticket  
- Fare  
- Cabin  
- Embarked  

---

## ✅ Steps Performed

### 1. Import and Explore the Dataset
- Loaded dataset using pandas.
- Explored structure using .info(), .describe(), and .head().
- Checked for missing values using .isnull().sum().
- Verified data types and used .value_counts() for categorical analysis.

### 2. Handle Missing Values
- Filled missing values in Age using *median*.
- Filled missing values in Embarked using *mode*.
- Dropped Cabin due to too many missing values.

### 3. Convert Categorical Features to Numerical
- Encoded the Sex column using *Label Encoding*.
- Used *One-Hot Encoding* for the Embarked column with pd.get_dummies().

### 4. Normalize / Standardize Numerical Features
- Applied *StandardScaler* to scale Age and Fare to standard normal distribution (mean=0, std=1).

### 5. Visualize and Remove Outliers
- Used *Seaborn boxplots* to visualize outliers in Age and Fare.
- Removed outliers using the *IQR (Interquartile Range)* method:
  - Retained data between Q1 - 1.5*IQR and Q3 + 1.5*IQR.

---

## 🛠 Libraries Used

```python
import pandas as pd  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.preprocessing import LabelEncoder, StandardScaler

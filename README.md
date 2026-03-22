#  Crop Yield Prediction using Machine Learning

## Overview
Agriculture is increasingly driven by data, yet predicting crop yield remains a complex challenge due to the influence of multiple factors such as weather conditions, soil properties, and environmental variability.

This project focuses on developing machine learning models to predict crop yield using agricultural datasets. The goal is to explore how data-driven approaches can support **intelligent decision-making in agriculture**, contributing to improved productivity and sustainable farming practices. 
The project is structured following a reproducible machine learning pipeline, reflecting best practices in research-oriented software development.

---

## 🎯 Objectives

- Develop predictive models for crop yield estimation  
- Analyze the impact of environmental and agricultural features  
- Compare performance of different machine learning algorithms  
- Explore the role of data-driven approaches in agricultural decision support systems  

---

## 📊 Dataset

The dataset used in this project includes agricultural and environmental features such as:

- Temperature  
- humidity  
- ph  
- water availability  
- seaon
- label 

** **Dataset source**: 
https://www.kaggle.com/datasets/rishabhrathore055/datas  

---

## ⚙️ Methodology

The project follows a standard machine learning pipeline:

### 1. Data Preprocessing
- Handling missing values  
- Feature selection and transformation  
- Data normalization  

### 2. Exploratory Data Analysis (EDA)
- Feature distribution analysis  
- Correlation analysis  
- Visualization of relationships between variables  

### 3. Model Development

The following models were implemented and compared:

- Linear Regression
- Logistic Regression
- Support Vector Machine
- Random Forest Classifier   
  
### 4. Model Evaluation

Models were evaluated using:

- Accuracy
- Confussion Matrix
- R² Score  

---

## 📈 Results

The machine learning models demonstrate the ability to capture relationships between environmental factors and crop yield.

Key observations:

- Tree-based models (e.g., Random Forest) performed better for non-linear relationships  
- Environmental variables such as humidity, ph, water availability, seaon, and temperature significantly impact crop_prediction  
- Model performance improves with better feature engineering and data quality  

---

## 🧠 Research Perspective

This project explores how machine learning can be applied to agricultural datasets to support **data-driven farming decisions**.

It highlights challenges such as:

- Heterogeneous agricultural data sources  
- Data quality and variability  
- Need for integrating multi-source datasets (weather, soil, sensor data)  

This work can be extended toward **AI-driven agricultural decision support systems** and **data integration frameworks for smart farming**.

---

## Future Work

- Integration of multi-source datasets (weather APIs, IoT sensor data)  
- Use of deep learning models for improved prediction  
- Development of real-time prediction systems  
- Deployment as a web-based decision support tool (e.g., Streamlit)  
- Incorporation of satellite data for crop monitoring  

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib / Seaborn  
- Jupyter Notebook  

---

## 📂 Project Structure
crop_yield_prediction/
│
├── data/ # Dataset files
├── notebooks/ # Jupyter notebooks
├── src/ # Source code 
├── models/ # Saved models
├── results/ # Output results and visualizations
├── README.md
└── requirements.txt

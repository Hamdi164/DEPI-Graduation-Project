# Microsoft Malware Prediction - Graduation Project
This repository contains the work for the Microsoft Malware Prediction project, completed as part of my training at DEPI. The project was based on a Kaggle competition and focused on predicting whether a Windows machine is likely to be infected by various families of malware using telemetry data.

### Project Overview
The goal of this project was to build a predictive model to determine the probability of malware infection on a Windows machine. The dataset provided for the competition contained telemetry data with over 8.9 million rows and 83 columns. Key highlights of the project include data analysis, feature engineering, model training, and deployment through a FastAPI app.

### Features
1. **Data Analysis**
Conducted extensive exploratory data analysis (EDA) to understand patterns and relationships within the data. Handled missing values, outliers, and data imbalances.

2. **Model Development**
Implemented machine learning algorithms, including gradient boosting and ensemble models. Performed hyperparameter tuning to optimize model performance. Evaluated models using metrics such as AUC-ROC and accuracy.

3. **Deployment**
Developed a FastAPI app to serve the trained model for real-time predictions. Designed RESTful endpoints to handle input data and return predictions.

4. **Documentation**
Comprehensive project documentation, including steps for data preprocessing, model training, and deployment. Instructions for reproducing the results and using the FastAPI app.

### Repository Structure
Microsoft-Malware-Prediction/

### Getting Started
Prerequisites Python 3.8 or higher
Required libraries listed in requirements.txt

### Acknowledgments
Kaggle for hosting the Microsoft Malware Prediction competition.
DEPI for guidance and training throughout the project.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

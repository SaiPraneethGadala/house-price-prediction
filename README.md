# 🏠 Hyderabad House Price Prediction using Machine Learning

## 📌 Project Overview

A Machine Learning web application that predicts house prices across different Hyderabad localities using **Linear Regression**. The project includes data generation, model training, and a user-friendly **Streamlit** web interface for real-time predictions.

---

## 🚀 Live Demo

**Streamlit App:**
https://house-price-prediction-b8wqxcivwdy32x6uundvey.streamlit.app/

---

## ✨ Features

* Predict house prices instantly
* Select Hyderabad locality
* Input area (sq ft), BHK, bathrooms, and property age
* Interactive Streamlit interface
* Machine Learning prediction using Scikit-learn Pipeline
* Responsive and easy-to-use UI

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* Pandas
* NumPy
* Streamlit
* Joblib

---

## 📂 Project Structure

```text
house-price-prediction/
│── app.py
│── train_model.py
│── generate_data.py
│── requirements.txt
│── README.md
│── data/
│   └── house_prices.csv
│── model/
│   └── house_price_model.joblib
```

---

## 📊 Dataset Features

* **Area** – Hyderabad locality
* **Sqft** – Built-up area
* **BHK** – Number of bedrooms
* **Bathrooms** – Number of bathrooms
* **Age (Years)** – Property age
* **Target** – House Price (Lakhs)

---

## 🤖 Machine Learning Workflow

1. Generate synthetic housing dataset
2. Data preprocessing
3. One-Hot Encoding for categorical features
4. Standard Scaling for numerical features
5. Train Linear Regression model
6. Save trained model using Joblib
7. Deploy prediction app with Streamlit

---

## 📈 Model Performance

| Metric    | Value             |
| --------- | ----------------- |
| Algorithm | Linear Regression |
| R² Score  | ~0.95             |
| MAE       | ~10.8 Lakhs       |

---

## ⚙️ Installation

```bash
git clone https://github.com/SaiPraneethGadala/house-price-prediction.git
cd house-price-prediction

python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

---

## 💡 Future Improvements

* Train using real Hyderabad housing data
* Add advanced ML models (Random Forest, XGBoost)
* Integrate interactive maps
* Improve prediction accuracy
* Deploy with Docker and CI/CD

---

## 👨‍💻 Developed By

**Sai Praneeth Gadala**

* GitHub: https://github.com/SaiPraneethGadala
* LinkedIn: https://linkedin.com/in/saipraneethgadala

---

**Note:** This project is built for educational and portfolio purposes. The current dataset is synthetic and intended to demonstrate the end-to-end Machine Learning workflow.

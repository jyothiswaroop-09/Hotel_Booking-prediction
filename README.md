# Data Science Project #

# 🏨 Hotel Booking Cancellation Prediction using Machine Learning

This project uses machine learning techniques to predict whether a hotel booking will be canceled, based on customer and booking attributes. The goal is to help hotels manage resources, reduce cancellations, and improve forecasting accuracy.

---

## 📌 Project Objective

- **Goal**: Predict if a hotel booking will be canceled (1) or not (0).
- **Dataset**: [Hotel Booking Demand Dataset](https://www.sciencedirect.com/science/article/pii/S2352340918315191)
- **Type**: Supervised Binary Classification
- **Tools**: Python, Scikit-learn, XGBoost, CatBoost, Pandas, SMOTE, Streamlit
- **Interface**: Simple form-based web interface
- **Deployment**: Localhost (Flask backend assumed)

---

## 📁 Project Structure
```
hotel-booking-cancellation/
├── artifacts/ # Preprocessed data, saved models
├── data/ # Original dataset (CSV)
├── notebooks/ # vs-code for EDA & modeling
├── src/ # Source code
│ ├── data_ingestion.py
│ ├── data_transformation.py
│ ├── model_trainer.py
│ └── utils.py
│ └── logger.py
│ └── exception.py
├── app/ # Streamlit prediction app
│ └── app.py
├── main.py # Pipeline trigger script
├── requirements.txt
├── README.md
```

---

## 📊 Features Used

Key input features include:

- `lead_time`
- `arrival_date_month`
- `adults`, `children`, `babies`
- `meal`, `market_segment`, `distribution_channel`
- `previous_cancellations`, `booking_changes`
- `deposit_type`
- `days_in_waiting_list`
- `customer_type`
- `adr` (Average Daily Rate)
- `required_car_parking_spaces`
- `total_of_special_requests`

**Target variable**: `is_canceled` (0 = Not Canceled, 1 = Canceled)

---

## 🧪 Data Preprocessing

- Handled missing values
- Removed outliers using **IQR method**
- Merged `children` and `babies` into `total_kids`
- Dropped irrelevant/redundant features
- Applied **Label Encoding** and **One-Hot Encoding**
- Balanced data using **SMOTE**
- Feature scaling using **StandardScaler**

Preprocessing pipelines and plots (QQ plots) saved in `artifacts/`.

---

## 🤖 Model Training

### Models Used:
- Random Forest
- XGBoost
- CatBoost
- Logistic Regression
- Support Vector Classifier

### Process:
- Trained using `GridSearchCV` for hyperparameter tuning
- Evaluated on accuracy and F1-score
- Saved best model in `.pkl` format

---

## 📉 Evaluation Metrics

- Accuracy
- Precision, Recall
- F1 Score
- ROC AUC Score
- Confusion Matrix

---

## 💻 How to Run
```
1️⃣ Clone the Repository
git clone https://github.com/jyothiswaroop-09/Hotel_Booking-prediction.git
cd hotel-booking-cancellation
2️⃣ Create & Activate Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run Training Pipeline
python main.py
5️⃣ Launch  App
python app.py
```

## 👨‍💻 Author
Jyothi Swaroop
GitHub: jyothiswaroop-09
Email: swaroop.motupalii@gmail.com


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

data = pd.read_csv('Train-data.csv')
data = data[['Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Mileage', 'Engine', 'Power', 'Seats', 'Price']]
data['Mileage'] = data['Mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '').astype(float)
data['Engine'] = data['Engine'].str.replace(' CC', '').astype(float)
data['Power'] = data['Power'].str.replace(' bhp', '').str.replace('null', '0').astype(float)
data.dropna(subset=['Mileage', 'Engine', 'Power'], inplace=True)

label_encoders = {}
for column in ['Fuel_Type', 'Transmission']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
X = data.drop('Price', axis=1)
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Điền giá trị thiếu trong tập huấn luyện và tập kiểm tra
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_train.mean(), inplace=True)


st.title("Dự đoán giá xe cũ")
st.sidebar.header("Nhập thông tin xe")
# Khai báo input_data dựa trên dữ liệu nhập từ sidebar
year = st.sidebar.number_input("Năm sản xuất", min_value=1900, max_value=2023, value=2020)
kilometers_driven = st.sidebar.number_input("Số kilomet đã đi", min_value=1, value=10000)
fuel_type = st.sidebar.selectbox("Loại nhiên liệu", label_encoders['Fuel_Type'].classes_)
transmission = st.sidebar.selectbox("Hộp số", label_encoders['Transmission'].classes_)
mileage = st.sidebar.number_input("Mileage (km/lít)", min_value=1, value=15)
engine = st.sidebar.number_input("Dung tích động cơ (cc)", min_value=1, value=1500)
power = st.sidebar.number_input("Công suất (bhp)", min_value=1, value=100)
seats = st.sidebar.number_input("Số chỗ ngồi", min_value=1, value=5)

input_data = pd.DataFrame({
    'Year': [year],
    'Kilometers_Driven': [kilometers_driven],
    'Fuel_Type': [label_encoders['Fuel_Type'].transform([fuel_type])[0]],
    'Transmission': [label_encoders['Transmission'].transform([transmission])[0]],
    'Mileage': [mileage],
    'Engine': [engine],
    'Power': [power],
    'Seats': [seats]
})

# Điền giá trị thiếu trong input_data
input_data.fillna(X_train.mean(), inplace=True)

print(input_data)

model = XGBRegressor(n_estimators=1000, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_pred, y_test)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")

predicted_price = model.predict(input_data)[0]
st.write(f"Dự đoán giá xe: {predicted_price:.2f} Lakh")

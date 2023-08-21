from data_processor import CarDataProcessor
from model_trainer import CarModelTrainer
from visualization import CarVisualizer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd

if __name__ == "__main__":
    data_path = "predic_price_car\Train-data.csv"

    data_processor = CarDataProcessor(data_path)
    data_processor.load_data()
    data_processor.preprocess_data()
    X_train, X_test, y_train, y_test = data_processor.split_data()

    model_trainer = CarModelTrainer(X_train.columns)
    model_trainer.train_model(X_train, y_train)
    r2 = model_trainer.evaluate_model(X_test, y_test)
    print(f'R^2: {r2}')

    new_car_data = {
        'Year': [2020],
        'Kilometers_Driven': [15000],
        'Fuel_Type': ['Petrol'],
        'Transmission': ['Manual'],
        'Owner_Type': ['First'],
        'Mileage': [20.0],
        'Engine': [1500],
        'Power': [100],
        'Seats': [5]
    }

    new_car_df = pd.DataFrame(new_car_data)
    
    # Chuyển đổi các cột Fuel_Type, Transmission, Owner_Type sang dạng số nguyên
    label_encoder = LabelEncoder()
    new_car_df['Fuel_Type'] = label_encoder.fit_transform(new_car_df['Fuel_Type'])
    new_car_df['Transmission'] = label_encoder.fit_transform(new_car_df['Transmission'])
    new_car_df['Owner_Type'] = label_encoder.fit_transform(new_car_df['Owner_Type'])

    new_car_dmatrix = xgb.DMatrix(new_car_df)
    predicted_price = model_trainer.predict_price(new_car_dmatrix)

    visualizer = CarVisualizer()
    visualizer.plot_actual_vs_predicted(y_test, model_trainer.model.predict(xgb.DMatrix(X_test)))
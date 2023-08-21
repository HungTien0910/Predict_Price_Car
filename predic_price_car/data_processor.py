import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class CarDataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.features = None
        self.target = None

    def load_data(self):
        self.data = pd.read_csv(self.data_path)

    def preprocess_data(self):
        self.data = self.data.drop(['Name', 'New_Price'], axis=1)
        self.data['Mileage'] = self.data['Mileage'].str.extract('(\d+\.\d+)').astype(float)
        self.data['Engine'] = self.data['Engine'].str.extract('(\d+)').astype(float)
        self.data['Power'] = self.data['Power'].str.extract('(\d+)').astype(float)

        label_encoder = LabelEncoder()
        self.data['Location'] = label_encoder.fit_transform(self.data['Location'])
        self.data['Fuel_Type'] = label_encoder.fit_transform(self.data['Fuel_Type'])
        self.data['Transmission'] = label_encoder.fit_transform(self.data['Transmission'])
        self.data['Owner_Type'] = label_encoder.fit_transform(self.data['Owner_Type'])

        self.features = self.data.drop('Price', axis=1)
        self.target = self.data['Price']

    def split_data(self, test_size=0.2, random_state=42):
        return train_test_split(self.features, self.target, test_size=test_size, random_state=random_state)

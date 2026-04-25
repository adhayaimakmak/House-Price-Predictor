import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
class DataLoader:
    def __init__(self,data):
        try:
            self.df = pd.read_csv(data)
        except Exception as e:
            print(f"Error initializing DataLoader: {e}")
            self.df = None

    def get_data(self):
        return self.df
    
    def get_shape(self):
        return f"Shape: {self.df.shape}"
    
    def get_dtypes(self):
        return self.df.dtypes

    def is_missing(self):
        missing_values = []
        for column in self.df.columns:
            missing_count = self.df[column].isnull().sum()
            if missing_count > 0:
                missing_values.append(f"Missing: {column} → {missing_count} rows")
        return '\n'.join(missing_values)
    
    def target_range(self, target_column):
        return f"Range: {self.df[target_column].min()} - {self.df[target_column].max()}"


class EDAAnalyzer:
    def __init__(self, data):
        self.df = data
        
    def split_features_target(self, target_column):
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        return X, y 
    
    def create_new_feature(self, df):
        df['rooms_per_household'] = df['total_rooms'] / df['households'].replace(0, np.nan)
        df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms'].replace(0, np.nan)
    
        df = df.drop(columns=['total_rooms', 'total_bedrooms', 'population', 'households'])
        return df
        
    def Categorize(self, df):
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = df.select_dtypes(include=['object', 'str']).columns

        return numeric_features, categorical_features
    
    def train_test_split_data(self,X,y,test_size=0.2,random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    def processed_fit_Transform(self, proprocessor, X_train, X_test):
        X_train_processed = proprocessor.fit_transform(X_train)
        X_test_processed = proprocessor.transform(X_test)
        return X_train_processed, X_test_processed
    
    def create_preprocessing_pipeline(self, numeric_features, categorical_features):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # แปลงข้อความเป็น One-Hot 
        ])
        
        preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
        )
        return preprocessor
        
    def plot_correlation_heatmap(self):
        plt.figure(figsize=(12, 10))
        numeric_df = self.df.select_dtypes(include='number')  
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.savefig('graphs/correlation_heatmap.png')
        plt.show()
        
class  ModelTrainer:
    def __int__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


def main():
    #ส่วนของ DataLoader 
    path = r'housing.csv'
    data_loader = DataLoader(path)
    print(data_loader.get_shape())
    print(data_loader.get_dtypes())
    print(data_loader.is_missing())
    print(data_loader.target_range('median_house_value'))
    data = data_loader.get_data()
    print(data.head())
    
    #ส่วนของการวิเคราะห์ EDA
    
    eda = EDAAnalyzer(data)
    #eda.plot_correlation_heatmap()#สร้างกราฟความสัมพันธ์ระหว่างตัวแปรเชิงตัวเลขลบคอมเมนต์ออกเมื่อรันครั้งแรกและปิดเพื่อดูกราฟความสัมพันธ์ระหว่างตัวแปรเชิงตัวเลขในครั้งถัดไป
    target_column = 'median_house_value'
    X, y = eda.split_features_target(target_column)
    X = eda.create_new_feature(X)
    
    X_train, X_test, y_train, y_test = eda.train_test_split_data(X,y)
    
    numeric_features,categorical_features = eda.Categorize(X)
    
    preprocessor = eda.create_preprocessing_pipeline(numeric_features, categorical_features)
    print(preprocessor)
    X_train_processed , X_test_processed = eda.processed_fit_Transform(preprocessor, X_train, X_test)
    
if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
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
    
    def manage_missing(self):
        for column in self.df.columns:
            if self.df[column].isnull().sum()>0:
                if self.df[column].dtype in ['float64', 'int64']:
                    self.df[column].fillna(self.df[column].median(), inplace=True)
                else:
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
            else:
                continue
        return self.df
        
    def Categorize(self):
        numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.df.select_dtypes(include=['object']).columns
        return numeric_features, categorical_features
    
    def create_preprocessing_pipeline(self, numeric_features, categorical_features):
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
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
    eda.manage_missing()
    eda.plot_correlation_heatmap()

if __name__ == "__main__":
    main()
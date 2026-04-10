import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib as jb

class DataLoader:
    def __init__(self,source:str):
        try:
            self.source = source
            self.df = None
        except Exception as e:
            print(f"Error initializing DataLoader: {e}")
        
    def load(self):
        self.df = pd.read_csv(self.source)
        return self.df
    
    def summary(self):
        return self.df.describe()
    
class EDAAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
       
    def __str__(self):
        return f"EDAAnalyzer with {self.df.shape[0]} rows and {self.df.shape[1]} columns\nData Types:\n{self.df.dtypes}"
      
    def missing_report(self) -> pd.Series:
        return self.df.isnull().sum() 
    
    def Target_range(self, target_column: str) -> str:
        if target_column not in self.df.columns:
            raise ValueError(f"Column '{target_column}' not found in DataFrame.")
        
        if not np.issubdtype(self.df[target_column].dtype, np.number):
            raise TypeError(f"Column '{target_column}' must be numeric.")
        
        min_val = self.df[target_column].min()
        max_val = self.df[target_column].max()
        
        # จัดรูปแบบตัวเลขให้มี comma และ $
        return f"Target range: ${min_val:,.0f}–${max_val:,.0f}"
    
    def plot_correlation(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()   
        
class processing:
    def __init__(self, target: str):
        self.target = target
        self.pipeline = None

    def build_pipeline(self):
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        return self

    def fit_transform(self, X):
        return self.pipeline.fit_transform(X)
        
    def manage_missing_values(self, column_name):
        if self.df[column_name].dtype in ['float64', 'int64']:
            median_val = self.df[column_name].median()
            return self.df[column_name].fillna(median_val, inplace=True)
        else:
            self.df[column_name] = self.df[column_name].fillna(self.df[column_name].mode()[0])
            return self.df[column_name]
    
    def summarize(self):
        return self.df.describe()
    
def main():
    data_loader = DataLoader(r'housing.csv')
    data_loader.load()
    print(data_loader.summary())
    eda = EDAAnalyzer(data_loader.df)
    print(eda)
    print(eda.missing_report())
    print(eda.Target_range('median_house_value'))  # Replace with actual target column name
    
    processor = processing(data_loader.df)
    processor.manage_missing_values('total_bedrooms')  # Replace with actual column name
    print(processor.summarize())
if __name__ == "__main__":
    main()
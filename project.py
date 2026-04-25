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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

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
        categorical_features = df.select_dtypes(include=['object']).columns

        return numeric_features, categorical_features
    
    def train_test_split_data(self,X,y,test_size=0.2,random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    
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
    
    def processed_fit_Transform(self, proprocessor, X_train, X_test):
        X_train_processed = proprocessor.fit_transform(X_train)
        X_test_processed = proprocessor.transform(X_test)
        return X_train_processed, X_test_processed
    
    def processed_Transform_new_X(self,proprocessor,X_new):
        new_proprocessor = proprocessor.transform(X_new)
        return new_proprocessor
        
    def plot_correlation_heatmap(self):
        plt.figure(figsize=(12, 10))
        numeric_df = self.df.select_dtypes(include='number')  
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.savefig('graphs/correlation_heatmap.png')
        plt.show()
        
class  ModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test, X_new=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test       
        self.X_new = X_new
    def train_linear_regression(self):
        self.model_linear_regression = LinearRegression()
        self.model_linear_regression.fit(self.X_train, self.y_train)
        return self.model_linear_regression     
    
    def train_RandomForest_Regressor(self):
        self.model_random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_random_forest.fit(self.X_train, self.y_train)
        return self.model_random_forest   
    
    def evaluate_model(self, model):
        y_pred = model.predict(self.X_test)
        return y_pred
    
    def Indicators_model(self, y_pred):
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return rmse, mae, r2
        
    
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
    #eda.plot_correlation_heatmap()#สร้างกราฟความสัมพันธ์ระหว่างตัวแปรเชิงตัวเลขลบคอมเมนต์ออกเมื่อรัน
    #ครั้งแรกและปิดเพื่อดูกราฟความสัมพันธ์ระหว่างตัวแปรเชิงตัวเลขในครั้ง ถัดไป
    target_column = 'median_house_value'
    X, y = eda.split_features_target(target_column)
    X = eda.create_new_feature(X)
    print(X.head())
    X_train, X_test, y_train, y_test = eda.train_test_split_data(X,y)
    
    numeric_features,categorical_features = eda.Categorize(X)
    
    preprocessor = eda.create_preprocessing_pipeline(numeric_features, categorical_features)
    
    X_train_processed , X_test_processed = eda.processed_fit_Transform(preprocessor, X_train, X_test)
    
    #ส่วนของ ModelTrainer
    model_trainer = ModelTrainer(X_train_processed, y_train, X_test_processed, y_test)
    model_linear_regression = model_trainer.train_linear_regression() #เทรนโมเดล
    model_random_forest = model_trainer.train_RandomForest_Regressor()
    
    evaluate_model_linear_regression = model_trainer.evaluate_model(model_linear_regression) #ทำนาย
    evaluate_model_random_forest = model_trainer.evaluate_model(model_random_forest)
    
    trainrmse_lr, trainmae_lr, trainr2_lr = model_trainer.Indicators_model(evaluate_model_linear_regression)
    trainrmse_rf, trainmae_rf, trainr2_rf = model_trainer.Indicators_model(evaluate_model_random_forest)
    print(f"Linear Regression - RMSE: {trainrmse_lr:<10.2f}, MAE: {trainmae_lr:<10.2f}, R2 Score: {trainr2_lr:<10.2f}")
    print(f"Random Forest Regressor - RMSE: {trainrmse_rf:<10.2f}, MAE: {trainmae_rf:<10.2f}, R2 Score: {trainr2_rf:<10.2f}")
    #ลองเพิ่มข้อมูลใหม่
    new_data = pd.DataFrame([{
    "longitude": -122.05,
    "latitude": 37.37,
    "housing_median_age": 30,
    "median_income":3.5,
    "total_rooms": 2000,
    "total_bedrooms": 500,
    "population": 800,
    "households": 300,
    "ocean_proximity":"NEAR BAY"
}])
    new_data_eda = EDAAnalyzer(new_data)
    X_new = new_data_eda.create_new_feature(new_data)
    new_proprocessor = new_data_eda.processed_Transform_new_X(preprocessor,X_new)
    model_trainer.X_new = new_proprocessor
    new_proprocessor_lr = model_linear_regression.predict(model_trainer.X_new)
    new_proprocessor_rf = model_random_forest.predict(model_trainer.X_new)
    print(f"Predicted median house value (Linear Regression): {new_proprocessor_lr[0]:.2f}")
    print(f"Predicted median house value (Random Forest Regressor): {new_proprocessor_rf[0]:.2f}")
    
if __name__ == "__main__":
    main()
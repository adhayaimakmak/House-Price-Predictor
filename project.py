import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    
def main():
    data_loader = DataLoader(r'housing.csv')
    data_loader.load()
    print(data_loader.summary())
if __name__ == "__main__":
    main()
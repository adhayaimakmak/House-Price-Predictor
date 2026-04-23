import pandas as pd
class DataLoader:
    def __init__(self,data):
        try:
            self.df = pd.read_csv(data)
        except Exception as e:
            print(f"Error initializing DataLoader: {e}")
            self.data = None
    def get_data(self):
        return self.df

def main():
    
    path = r'housing.csv'
    data_loader = DataLoader(path)
    data = data_loader.get_data()
    print(data.head())
    
if __name__ == "__main__":
    main()
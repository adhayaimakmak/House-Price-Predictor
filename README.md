# House Price Predictor — โปรเจกต์ทำนายราคาบ้าน
__โจทย์__
***
สร้างระบบทำนายราคาบ้านจาก Dataset สาธารณะ (California Housing หรือ Boston Housing) โดยใช้ OOP สร้าง Pipeline ตั้งแต่โหลดข้อมูล → ทำ EDA → Preprocessing → Train Model → ประเมินผล → Export Report

## ข้อจำกัดโปรเจกต์
1. __ต้องใช้ OOP__
   แต่ละขั้นตอนต้องเป็น Class เช่น DataLoader, EDAAnalyzer, ModelTrainer
2. __ไม่ใช้ AutoML__
   ต้องเข้าใจแต่ละขั้นตอน ห้ามใช้ auto-sklearn หรือ TPOT แบบ black-bo
3. __มี Evaluation ครบ__
   ต้องรายงาน MAE, RMSE, R² พร้อม Visualization อย่างน้อย 3 กราฟ

## ขั้นตอนการทำโปรเจกต์
1. Setup Environment ติดตั้ง pip install pandas numpy scikit-learn matplotlib seaborn joblib แล้วเปิด Jupyter Notebook

2. โหลด Dataset ใช้ sklearn.datasets.fetch_california_housing() หรือดาวน์โหลดจาก Kaggle (ฟรี) สร้าง DataLoader class

3. Exploratory Data Analysis (EDA) วิเคราะห์การแจกแจง, ค่า missing, correlation matrix ผ่าน EDAAnalyzer class

4. Preprocessing แก้ missing values, scale features ด้วย Preprocessor + sklearn.Pipeline

5. Train & Evaluate ทดลองอย่างน้อย 2 model (Linear Regression vs Random Forest) เปรียบเทียบด้วย MAE, RMSE, R²

6. Visualize Results วาด: Actual vs Predicted plot, Residual plot, Feature Importance bar chart

7. Save & Document บันทึก model ด้วย joblib, เขียน README อธิบายโปรเจกต์ใน GitHub

## เครื่องมือที่ต้องใช้และการติดตั้ง

`pip install pandas numpy scikit-learn matplotlib seaborn joblib `

- IDE: แล้วแต่ศรัทธา เช่น jupyter notebook, Google colab, vscode เป็นต้น

## Input → Output ที่ต้องการ
1. โหลดข้อมูล
```bash
INPUT (CSV)
longitude,latitude,age,rooms,bedrooms,price
-122.23,37.88,41,880,129,452600
-122.22,37.86,21,7099,1106,358500
-122.24,37.85,52,1467,190,352100
```
```bash
OUTPUT (summary)
Shape: (20640, 9)
Missing: bedrooms → 207 rows
Dtypes: float64 x8, object x1
Target range: $14,999–$500,001
```
2. EDA Report
```bash
INPUT
eda = EDAAnalyzer(df)
eda.missing_report()
eda.plot_correlation()
```
```bash
OUTPUT
bedrooms      207
total_rooms     0
ocean_prox      0
dtype: int64
→ Correlation heatmap saved
```
3. ผลลัพธื model
```bash
INPUT
trainer = ModelTrainer("rf")
trainer.train(X_train, y_train)
trainer.evaluate(X_test, y_test)
```
```bash
OUTPUT
{
  "MAE":  32450.12,
  "RMSE": 49820.55,
  "R2":   0.8147
}
→ model.pkl saved (12.3 MB)
```

4. 
```bash
INPUT (dict)
{
  "longitude": -122.05,
  "latitude": 37.37,
  "housing_median_age": 30,
  "total_rooms": 2000,
  "population": 800
}
```

```bash
OUTPUT (prediction)
Predicted price:
  $287,450

Confidence (±1σ):
  $238,200 – $336,700
```

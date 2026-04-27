# 🏠 House Price Predictor

ระบบทำนายราคาบ้านโดยใช้ Machine Learning

## 📋 สารบัญ
- [คำอธิบายโปรเจกต์](#-คำอธิบายโปรเจกต์)
- [วัตถุประสงค์](#-วัตถุประสงค์)
- [ข้อมูลและแหล่งที่มา](#-ข้อมูลและแหล่งที่มา)
- [ติดตั้งและสภาพแวดล้อม](#-ติดตั้งและสภาพแวดล้อม)
- [โครงสร้างไฟล์](#-โครงสร้างไฟล์)
- [วิธีการใช้งาน](#-วิธีการใช้งาน)
- [โมเดล Machine Learning](#-โมเดล-machine-learning)
- [ผลลัพธ์และการประเมิน](#-ผลลัพธ์และการประเมิน)
- [โครงสร้างข้อมูล](#-โครงสร้างข้อมูล)
- [ผู้พัฒนา](#-ผู้พัฒนา)

## 📖 คำอธิบายโปรเจกต์

โปรเจกต์นี้สร้างขึ้นเพื่อทำนายราคาบ้านจากชุดข้อมูล California Housing Dataset โดยใช้หลักการ Object-Oriented Programming (OOP) เพื่อให้โค้ดเป็นระเบียบ นำกลับมาใช้ได้ และง่ายต่อการบำรุงรักษา

โปรเจกต์นี้ครอบคลุมทั้งขั้นตอนการวิเคราะห์ข้อมูล (EDA) การเตรียมข้อมูล (Preprocessing) การฝึกสอนโมเดล (Model Training) และการประเมินผล (Model Evaluation)

## 🎯 วัตถุประสงค์

1. **เข้าใจขั้นตอน Machine Learning Pipeline** - จากการโหลดข้อมูลจนถึงการทำนาย
2. **ฝึกฝนการใช้ OOP ในงานวิทยาศาสตร์ข้อมูล** - สร้าง Class สำหรับ DataLoader, EDAAnalyzer, และ ModelTrainer
3. **เทียบเทียบโมเดล** - เปรียบเทียบประสิทธิภาพของ Linear Regression กับ Random Forest
4. **สร้างรายงานและการแสดงผลข้อมูล** - ใช้ Visualization เพื่ออธิบายผลลัพธ์

## 📊 ข้อมูลและแหล่งที่มา

**ชื่อชุดข้อมูล:** California Housing Dataset

**จำนวนตัวอย่าง:** 20,640 ตัวอย่าง

**จำนวนฟีเจอร์:** 9 ฟีเจอร์

**ตัวแปรเป้าหมาย:** median_house_value (ราคาบ้านมัธยฐาน)

**แหล่งที่มา:** scikit-learn datasets หรือ Kaggle (ฟรี)

### ฟีเจอร์ในชุดข้อมูล:
| ฟีเจอร์ | คำอธิบาย | ประเภท |
|--------|---------|--------|
| longitude | ลองจิจูด (พิกัด X) | float |
| latitude | ละติจูด (พิกัด Y) | float |
| housing_median_age | อายุค่ามัธยฐานของบ้าน | float |
| total_rooms | จำนวนห้องทั้งหมด | float |
| total_bedrooms | จำนวนห้องนอนทั้งหมด | float |
| population | ประชากร | float |
| households | จำนวนครัวเรือน | float |
| median_income | รายได้มัธยฐาน | float |
| median_house_value | **ราคาบ้านมัธยฐาน (Target)** | float |
| ocean_proximity | ความใกล้ชิดกับมหาสมุทร | object |

## 🔧 ติดตั้งและสภาพแวดล้อม

### ข้อกำหนดเบื้องต้น
- Python 3.8+
- pip (ตัวจัดการแพ็คเกจ Python)

### การติดตั้ง

1. **โคลนหรือดาวน์โหลดโปรเจกต์**
```bash
cd House_Price_Predictor
```

2. **ติดตั้งแพ็คเกจที่จำเป็น**
```bash
pip install -r requirements.txt
```

### แพ็คเกจที่ใช้
| แพ็คเกจ | เวอร์ชัน | วัตถุประสงค์ |
|--------|---------|----------|
| pandas | 2.2.0 | การจัดการและวิเคราะห์ข้อมูล |
| numpy | 1.26.4 | การคำนวณตัวเลขและพีชคณิต |
| scikit-learn | 1.4.2 | Machine Learning algorithms |
| matplotlib | 3.8.4 | การเขียนกราฟ 2D |
| seaborn | 0.13.1 | การเขียนกราฟแบบสถิติ |
| joblib | 1.3.2 | บันทึกและโหลดโมเดล |

## 📁 โครงสร้างไฟล์

```
House_Price_Predictor/
│
├── housing.csv                 # ชุดข้อมูล California Housing
├── project.py                  # โค้ดหลัก (DataLoader, EDAAnalyzer)
├── README.md                   # ไฟล์คำอธิบายนี้
├── Project_details.md          # รายละเอียดโปรเจกต์
├── requirements.txt            # รายชื่อแพ็คเกจที่ต้องการ
│
└── graphs/                     # โฟลเดอร์สำหรับบันทึกกราฟ
    ├── correlation_heatmap.png
    ├── actual_vs_predicted.png
    ├── residual_plot.png
    └── feature_importance.png
```

## 💻 วิธีการใช้งาน

### ขั้นตอนที่ 1: โหลดข้อมูล
```python
from project import DataLoader

loader = DataLoader('housing.csv')
df = loader.get_data()

print(loader.get_shape())
print(loader.is_missing())
print(loader.target_range('median_house_value'))
```

**ผลลัพธ์ที่คาดหวัง:**
```
Shape: (20640, 10)
Missing: total_bedrooms → 207 rows
Range: 14999.0 - 500001.0
```

### ขั้นตอนที่ 2: วิเคราะห์ข้อมูล (EDA)
```python
from project import EDAAnalyzer

eda = EDAAnalyzer(df)
X, y = eda.split_features_target('median_house_value')

X_processed = eda.create_new_feature(X.copy())
numeric_features, categorical_features = eda.Categorize(X_processed)
```

### ขั้นตอนที่ 3: แบ่งข้อมูลและเตรียมการ (Preprocessing)
```python
X_train, X_test, y_train, y_test = eda.train_test_split_data(X_processed, y)

preprocessor = eda.create_preprocessing_pipeline(numeric_features, categorical_features)
X_train_processed, X_test_processed = eda.processed_fit_Transform(preprocessor, X_train, X_test)
```

### ขั้นตอนที่ 4: ฝึกสอนโมเดล
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_processed, y_train)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_processed, y_train)
```

### ขั้นตอนที่ 5: ประเมินผล
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Linear Regression Evaluation
lr_pred = lr_model.predict(X_test_processed)
print(f"MAE: {mean_absolute_error(y_test, lr_pred):,.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred)):,.2f}")
print(f"R² Score: {r2_score(y_test, lr_pred):.4f}")

# Random Forest Evaluation
rf_pred = rf_model.predict(X_test_processed)
print(f"MAE: {mean_absolute_error(y_test, rf_pred):,.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):,.2f}")
print(f"R² Score: {r2_score(y_test, rf_pred):.4f}")
```

## 🤖 โมเดล Machine Learning

### 1. Linear Regression
**ข้อดี:**
- เรียบง่ายและเข้าใจได้ง่าย
- ฝึกสอนได้อย่างรวดเร็ว
- ใช้เน้อที่น้อย

**ข้อเสีย:**
- อาจไม่สามารถจับความสัมพันธ์ที่ซับซ้อนได้

### 2. Random Forest
**ข้อดี:**
- จับความสัมพันธ์ที่ไม่เป็นเชิงเส้นได้ดี
- ประสิทธิภาพดีกว่าปกติ
- ลดการ overfitting

**ข้อเสีย:**
- ช้ากว่า Linear Regression
- ใช้พื้นที่มากกว่า

## 📈 ผลลัพธ์และการประเมิน

### เมตริกการประเมิน

1. **Mean Absolute Error (MAE)**
   - ค่าผิดพลาดเฉลี่ยตามค่าสัมบูรณ์
   - หน่วย: ดอลลาร์

2. **Root Mean Squared Error (RMSE)**
   - รากที่สองของค่าเฉลี่ยของกำลังสองของข้อผิดพลาด
   - ให้ความสำคัญกับข้อผิดพลาดที่ใหญ่
   - หน่วย: ดอลลาร์

3. **R² Score (Coefficient of Determination)**
   - วัดสัดส่วนของความแปรปรวนที่อธิบายได้
   - ช่วง: 0 ถึง 1 (ยิ่งใกล้ 1 ยิ่งดี)

### การแสดงผลข้อมูล (Visualizations)

โปรเจกต์นี้สร้าง 4 กราฟหลัก:

1. **Correlation Heatmap** - แสดงความสัมพันธ์ระหว่างฟีเจอร์
2. **Actual vs Predicted** - เปรียบเทียบราคาที่แท้จริงกับราคาที่ทำนาย
3. **Residual Plot** - แสดงข้อผิดพลาดของการทำนาย
4. **Feature Importance** - แสดงความสำคัญของแต่ละฟีเจอร์ (สำหรับ Random Forest)

## 📊 โครงสร้างข้อมูล

### สรุปข้อมูล
- **ขนาด:** 20,640 แถว × 10 คอลัมน์
- **ค่า Missing:** total_bedrooms มี 207 ค่าที่หายไป
- **ช่วงราคา:** $14,999 - $500,001

### ประเภทข้อมูล
- **Numeric (float64):** 9 ฟีเจอร์
- **Categorical (object):** 1 ฟีเจอร์ (ocean_proximity)

### ฟีเจอร์ที่สร้างใหม่
- `rooms_per_household` = total_rooms / households
- `bedrooms_per_room` = total_bedrooms / total_rooms

## ✨ คุณสมบัติของโปรเจกต์

✅ ใช้ Object-Oriented Programming (OOP)
✅ การจัดการข้อมูลด้วย Pandas
✅ Exploratory Data Analysis (EDA)
✅ Data Preprocessing และ Feature Engineering
✅ Train-Test Split
✅ สร้าง Pipeline สำหรับ Preprocessing
✅ ฝึกสอนหลายโมเดล
✅ ประเมินผลด้วย MAE, RMSE, R²
✅ การแสดงผลข้อมูล (Visualization)
✅ บันทึกโมเดลด้วย joblib

## 🎓 สิ่งที่เรียนรู้

จากการทำโปรเจกต์นี้ คุณจะเรียนรู้:

- การออกแบบและสร้าง Machine Learning Pipeline
- การใช้ scikit-learn สำหรับ Data Preprocessing และ Modeling
- การใช้ OOP ในงาน Data Science
- การวิเคราะห์ข้อมูลและการแสดงผลข้อมูล
- การประเมินและเปรียบเทียบโมเดล ML
- Best Practices ในการจัดระเบียบโค้ด

## 📝 หมายเหตุ

- ข้อมูลขาดหาย (Missing Data) ใน total_bedrooms ได้รับการจัดการโดยใช้ SimpleImputer
- ฟีเจอร์ Categorical (ocean_proximity) ถูกแปลงเป็น One-Hot Encoding
- ฟีเจอร์ Numeric ได้รับการ Standardize ด้วย StandardScaler
- Random Forest ช่วยให้เห็นได้ว่าฟีเจอร์ใดมีความสำคัญที่สุด

## 👨‍💻 ผู้พัฒนา

- สร้างขึ้นเพื่อการศึกษาและฝึกฝนทักษะ Machine Learning
- ปรับปรุงและขยายได้ตามต้องการ

## 📄 ลิขสิทธิ์

โปรเจกต์นี้สำหรับการศึกษาและใช้งานส่วนตัว

---

**สร้างเมื่อ:** เมษายน 2026
**แล้วอัปเดตครั้งล่าสุด:** เมษายน 2026
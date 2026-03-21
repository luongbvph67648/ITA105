import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# =================BÀI 1
# TẠO DỮ LIỆU MẪU (Bỏ qua nếu bạn đã load file .csv)
data = {
    'area': [50, 60, 75, 80, 100, 120, 150, 300, 20, 500],
    'price': [1.5, 1.8, 2.2, 2.5, 3.0, 3.5, 4.5, 15.0, 0.5, 25.0]
}
df_housing = pd.DataFrame(data)

# 1. Nạp dữ liệu, kiểm tra shape, missing values [cite: 6]
print("Shape:", df_housing.shape)
print("Missing values:\n", df_housing.isnull().sum())

# 2. Thống kê mô tả [cite: 7]
print("\nThống kê mô tả:\n", df_housing.describe())

# 3. Vẽ boxplot cho từng biến numeric [cite: 9]
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(y=df_housing['area'])
plt.title('Boxplot Area')
plt.subplot(1, 2, 2)
sns.boxplot(y=df_housing['price'])
plt.title('Boxplot Price')
plt.show()

# 4. Vẽ scatterplot diện tích và giá [cite: 10]
plt.figure(figsize=(6, 4))
sns.scatterplot(x='area', y='price', data=df_housing)
plt.title('Area vs Price Scatter Plot')
plt.show()

# 5. Tính IQR và xác định ngoại lệ [cite: 11]
Q1 = df_housing.quantile(0.25)
Q3 = df_housing.quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ((df_housing < (Q1 - 1.5 * IQR)) | (df_housing > (Q3 + 1.5 * IQR))).sum()
print("\nSố lượng ngoại lệ theo IQR:\n", outliers_iqr)

# 6. Tính Z-score (|Z| > 3) [cite: 12, 13]
z_scores = np.abs(stats.zscore(df_housing.select_dtypes(include=[np.number])))
outliers_z = (z_scores > 3).sum()
print("\nSố lượng ngoại lệ theo Z-score:\n", outliers_z)

# 9. Xử lý ngoại lệ: Clip (Giới hạn giá trị) [cite: 17, 19]
df_housing_cleaned = df_housing.copy()
for col in ['area', 'price']:
    lower = Q1[col] - 1.5 * IQR[col]
    upper = Q3[col] + 1.5 * IQR[col]
    df_housing_cleaned[col] = np.clip(df_housing[col], lower, upper)

# 10. Vẽ lại boxplot sau xử lý [cite: 20]
sns.boxplot(data=df_housing_cleaned)
plt.title("Boxplot sau khi Clip ngoại lệ")
plt.show()
# =========================BÀI 2
# TẠO DỮ LIỆU MẪU SENSOR
dates = pd.date_range('2024-01-01', periods=100, freq='H')
df_iot = pd.DataFrame({
    'timestamp': dates,
    'temp': np.random.normal(25, 2, 100)
}, index=dates).drop(columns='timestamp')
df_iot.iloc[50] = 80  # Tạo điểm ngoại lệ giả lập 

# 2. Vẽ line plot [cite: 23]
df_iot['temp'].plot(title="Temperature over Time")
plt.show()

# 3. Rolling mean ± 3 x std [cite: 24]
window = 10
rolling_mean = df_iot['temp'].rolling(window=window).mean()
rolling_std = df_iot['temp'].rolling(window=window).std()
upper_bond = rolling_mean + (3 * rolling_std)
lower_bond = rolling_mean - (3 * rolling_std)

outliers_rolling = df_iot[(df_iot['temp'] > upper_bond) | (df_iot['temp'] < lower_bond)]
print("\nNgoại lệ phát hiện bởi Rolling Mean:\n", outliers_rolling)

# 7. Xử lý bằng Interpolation (Nội suy) [cite: 28]
df_iot_fixed = df_iot.copy()
# Giả sử ta coi các điểm > upper_bond là NaN để nội suy
df_iot_fixed.loc[outliers_rolling.index, 'temp'] = np.nan
df_iot_fixed['temp'] = df_iot_fixed['temp'].interpolate()
df_iot_fixed['temp'].plot(title="Sau khi xử lý bằng Interpolation")
plt.show()

# ================BÀI 3
# TẠO DỮ LIỆU MẪU E-COMMERCE
df_eco = pd.DataFrame({
    'price': [10, 20, 15, 1000, 12, 18, 5],
    'quantity': [1, 2, 1, 1, 100, 2, 1],
    'rating': [4.5, 4.0, 3.5, 5.0, 1.0, 4.2, 5.0]
})

#================== Bài 4: Multivariate Outlier (Diện tích + Giá) [cite: 45, 46]
# Sử dụng Scatter plot 2D để highlight [cite: 51]
plt.figure(figsize=(8, 6))
# Xác định ngoại lệ Multivariate đơn giản: các điểm nằm ngoài IQR của cả 2 biến
sns.scatterplot(x='area', y='price', data=df_housing, hue=((z_scores > 2).any(axis=1)), palette='coolwarm')
plt.title("Multivariate Outliers (Red points)")
plt.show()

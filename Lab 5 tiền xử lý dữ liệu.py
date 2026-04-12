# Bài 1
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Đọc dữ liệu
df1 = pd.read_csv('ITA105_Lab_5_Supermarket.csv')

# 2. Chuyển cột ngày về datetime và đặt làm index
df1['date'] = pd.to_datetime(df1['date'])
df1.set_index('date', inplace=True)

# 3. Kiểm tra và xử lý missing values (Dùng Forward Fill theo yêu cầu)
df1['revenue'] = df1['revenue'].ffill()

# 4. Tạo đặc trưng thời gian
df1['year'] = df1.index.year
df1['month'] = df1.index.month
df1['quarter'] = df1.index.quarter
df1['day_of_week'] = df1.index.dayofweek
df1['is_weekend'] = df1['day_of_week'].isin([5, 6])

# 5. Vẽ biểu đồ tổng doanh thu theo tháng
df1.resample('ME')['revenue'].sum().plot(kind='line', title='Tổng doanh thu theo tháng')
plt.show()

# 6. Phân tích Trend & Seasonality (Decomposition)
result = seasonal_decompose(df1['revenue'], model='additive', period=30)
result.plot()
plt.show()
# Bài 2
# 1. Đọc dữ liệu
df2 = pd.read_csv('ITA105_Lab_5_Web_traffic.csv')
df2['datetime'] = pd.to_datetime(df2['datetime'])
df2.set_index('datetime', inplace=True)

# 2. Đặt lại tần suất hourly và xử lý missing value bằng nội suy (Linear Interpolation)
df2 = df2.asfreq('h')
df2['visits'] = df2['visits'].interpolate(method='linear')

# 3. Tạo biến đặc trưng
df2['hour'] = df2.index.hour
df2['day_of_week'] = df2.index.dayofweek

# 4. Vẽ biểu đồ lưu lượng theo giờ để tìm Peak/Trough
df2.groupby('hour')['visits'].mean().plot(kind='bar', title='Lưu lượng trung bình theo giờ')
plt.show()
# Bài 3
# 1. Đọc và đặt index
df3 = pd.read_csv('ITA105_Lab_5_Stock.csv')
df3['date'] = pd.to_datetime(df3['date'])
df3.set_index('date', inplace=True)

# 2. Xử lý ngày nghỉ (Missing values) bằng Forward Fill
df3 = df3.ffill()

# 3. Tạo Rolling Mean 7 ngày và 30 ngày
df3['RM7'] = df3['close_price'].rolling(window=7).mean()
df3['RM30'] = df3['close_price'].rolling(window=30).mean()

# 4. Vẽ biểu đồ so sánh
df3[['close_price', 'RM7', 'RM30']].plot(figsize=(12,6), title='Giá cổ phiếu và Rolling Mean')
plt.show()
# Bài 4
# 1. Đọc dữ liệu
df4 = pd.read_csv('ITA105_Lab_5_Production.csv')
df4['week_start'] = pd.to_datetime(df4['week_start'])
df4.set_index('week_start', inplace=True)

# 2. Điền missing values
df4 = df4.ffill()

# 3. Tạo đặc trưng và phân tích mùa vụ theo quý
df4['quarter'] = df4.index.quarter
df4.groupby('quarter')['production'].mean().plot(kind='bar', title='Sản lượng trung bình theo quý')
plt.show()

# 4. Phân tích Decomposition (Sử dụng statsmodels)
# Vì dữ liệu theo tuần, period thường chọn là 52
result_prod = seasonal_decompose(df4['production'], model='additive', period=52)
result_prod.plot()
plt.show()
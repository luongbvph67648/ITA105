import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('ITA105_Lab_1.csv')

print("============= DỮ LIỆU BAN ĐẦU =============")
print(df)
print("\n")

# BÀI 1: KHÁM PHÁ DỮ LIỆU
print("============= BÀI 1: KHÁM PHÁ DỮ LIỆU =============")
# 1.1 Kiểm tra kích thước dữ liệu
print("1. Kích thước dữ liệu:")
print(f"- Số dòng: {df.shape[0]}")
print(f"- Số cột: {df.shape[1]}")

# 1.2 Xem thống kê mô tả của các cột số
print("\n2. Thống kê mô tả của các cột số:")
print(df.describe())

# 1.3 Kiểm tra giá trị thiếu trong các cột
print("\n3. Kiểm tra giá trị thiếu trong các cột:")
print(df.isnull().sum())
print("\n")


# BÀI 2: XỬ LÝ DỮ LIỆU THIẾU
print("============= BÀI 2: XỬ LÝ DỮ LIỆU THIẾU =============")
# 2.1 Dùng .isnull().sum() để phát hiện giá trị thiếu
print("1. Phát hiện giá trị thiếu (lại):")
print(df.isnull().sum())

# 2.2 Điền giá trị thiếu với mean/median/mode
df_filled = df.copy()
# Category là biến phân loại nên dùng mode (giá trị xuất hiện nhiều nhất)
mode_category = df_filled['Category'].mode()[0]
df_filled['Category'] = df_filled['Category'].fillna(mode_category)

# StockQuantity là biến số đếm nên có thể dùng median
median_stock = df_filled['StockQuantity'].median()
df_filled['StockQuantity'] = df_filled['StockQuantity'].fillna(median_stock)

print("\n2. Dữ liệu sau khi điền giá trị thiếu (fillna):")
print(df_filled[['Category', 'StockQuantity']])

# 2.3 So sánh kết quả với phương pháp dropna()
df_dropped = df.dropna()
print("\n3. Dữ liệu sau khi xóa giá trị thiếu (dropna):")
print(f"Kích thước ban đầu: {df.shape}")
print(f"Kích thước sau khi dropna: {df_dropped.shape}")
print(df_dropped[['Category', 'StockQuantity']])
print("\n")

# Cập nhật df bằng df_filled để làm tiếp
df = df_filled.copy()


# BÀI 3: XỬ LÝ DỮ LIỆU LỖI
print("============= BÀI 3: XỬ LÝ DỮ LIỆU LỖI =============")
# 3.1 Kiểm tra và xử lý các giá trị bất hợp lý trong cột Price và StockQuantity
# Giả sử Price phải >= 0 và < 10000 (loại bỏ giá trị outlier quá lớn như 1000000)
# StockQuantity phải >= 0
print("1. Xử lý giá trị bất hợp lý (Price < 0, Price quá lớn, StockQuantity < 0):")
# Thay thế giá trị lỗi bằng NaN
df.loc[(df['Price'] < 0) | (df['Price'] > 10000), 'Price'] = np.nan
df.loc[df['StockQuantity'] < 0, 'StockQuantity'] = np.nan

# Điền lại các giá trị này bằng median (hoặc mean tùy chọn)
df['Price'] = df['Price'].fillna(df['Price'].median())
df['StockQuantity'] = df['StockQuantity'].fillna(df['StockQuantity'].median())

print(df[['Price', 'StockQuantity']])

# 3.2 Lọc các giá trị không hợp lệ trong cột Rating (1 <= Rating <= 5)
print("\n2. Lọc các giá trị Rating hợp lệ (1 <= Rating <= 5):")
df = df[(df['Rating'] >= 1) & (df['Rating'] <= 5)].copy()
print(df[['Rating']])
print("\n")


# BÀI 4: LÀM MƯỢT DỮ LIỆU NHIỄU
print("============= BÀI 4: LÀM MƯỢT DỮ LIỆU NHIỄU =============")
# Dùng chuỗi thời gian là cột Price để làm mượt
time_series_price = df['Price'].reset_index(drop=True)

# 4.1 Áp dụng Moving Average để làm mượt dữ liệu cột Price
window_size = 5
smoothed_price = time_series_price.rolling(window=window_size).mean()

# 4.2 Vẽ biểu đồ line trước và sau khi làm mượt
plt.figure(figsize=(10, 5))
plt.plot(time_series_price, label='Dữ liệu gốc', alpha=0.5, linestyle='--')
plt.plot(smoothed_price, label=f'Dữ liệu làm mượt (MA, window={window_size})', color='red', linewidth=2)
plt.title('Làm mượt dữ liệu Price bằng Moving Average')
plt.xlabel('Thời gian / Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Lưu biểu đồ ra file thay vì show() để không chặn luồng chạy lệnh
plt.savefig('price_smoothing.png')
print("Đã vẽ biểu đồ và lưu thành file 'price_smoothing.png'")
print("\n")


# BÀI 5: CHUẨN HÓA DỮ LIỆU
print("============= BÀI 5: CHUẨN HÓA DỮ LIỆU =============")
# 5.1 Chuyển tất cả giá trị trong cột Category thành chữ thường và xóa khoảng trắng 2 đầu
df['Category'] = df['Category'].str.lower().str.strip()
print("1. Cột Category sau khi chuyển chữ thường:")
print(df[['Category']])

# 5.2 Loại bỏ ký tự thừa trong cột Description (chỉ giữ lại chữ cái, số và khoảng trắng)
print("\n2. Cột Description sau khi làm sạch ký tự thừa:")
df['Description'] = df['Description'].astype(str).str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
print(df[['Description']])

# 5.3 Chuyển đổi đơn vị giá từ USD sang VND (Tỷ giá: 1 USD = 25,000 VND)
print("\n3. Chuyển đổi đơn vị giá từ USD sang VND:")
exchange_rate = 25000
df['Price_VND'] = df['Price'] * exchange_rate
print(df[['Price', 'Price_VND']])
print("\n============= HOÀN THÀNH =============")
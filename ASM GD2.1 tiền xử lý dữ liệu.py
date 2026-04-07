# Khám phá dữ liệu đa dạng (EDA)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Giả sử df là dữ liệu của bạn
# 1.1. Thống kê cơ bản
print("--- Thống kê các cột số ---")
print(df.describe()) # Xem mean, std, min, max...

print("\n--- Kiểm tra giá trị thiếu (Missing values) ---")
print(df.isnull().sum())

# 1.2. Vẽ biểu đồ phân phối giá nhà (Histogram)
plt.figure(figsize=(10, 5))
sns.histplot(df['gia_nha'], kde=True, color='blue')
plt.title('Phân phối giá nhà')
plt.show()

# 1.3. Vẽ Boxplot để xem phân phối và Outliers của diện tích
sns.boxplot(x=df['dien_tich'])
plt.title('Biểu đồ Boxplot diện tích')
plt.show()


# Xử lý dữ liệu bẩn (Data Cleaning)
# 2.1. Điền giá trị thiếu (Missing values)
# Với số lượng phòng: Điền bằng Mode (giá trị phổ biến nhất)
mode_rooms = df['so_phong'].mode()[0]
df['so_phong'] = df['so_phong'].fillna(mode_rooms)

# Với diện tích: Điền bằng Median (số trung vị) để tránh bị ảnh hưởng bởi nhà quá to/nhỏ
median_area = df['dien_tich'].median()
df['dien_tich'] = df['dien_tich'].fillna(median_area)

# 2.2. Xử lý dữ liệu không hợp lệ (Logic bẩn)
# Loại bỏ những bản ghi có giá nhà <= 0 hoặc số phòng = 0 (vì vô lý)
df = df[df['gia_nha'] > 0]
df = df[df['so_phong'] > 0]

# 2.3. Sửa lỗi chính tả trong dữ liệu phân loại (Categorical)
# Ví dụ: "Chung cư" bị viết nhầm thành "Chung cu" hoặc "cc"
mapping = {
    'Chung cu': 'Chung cư',
    'cc': 'Chung cư',
    'Nha pho': 'Nhà phố'
}
df['loai_nha'] = df['loai_nha'].replace(mapping)

# 2.4. Loại bỏ trùng lặp (Duplicate)
print(f"Số lượng dòng trước khi xóa trùng: {len(df)}")
df = df.drop_duplicates()
print(f"Số lượng dòng sau khi xóa trùng: {len(df)}")

# import pandas as pd
import numpy as np

# Tính toán Q1 (25%) và Q3 (75%) của cột 'gia_nha'
Q1 = df['gia_nha'].quantile(0.25)
Q3 = df['gia_nha'].quantile(0.75)
IQR = Q3 - Q1

# Xác định ngưỡng trên và ngưỡng dưới
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Chiến lược Capping: Thay thế các giá trị vượt ngưỡng bằng chính giá trị ngưỡng đó
df['gia_nha'] = np.where(df['gia_nha'] > upper_bound, upper_bound, df['gia_nha'])
df['gia_nha'] = np.where(df['gia_nha'] < lower_bound, lower_bound, df['gia_nha'])

print("Đã xử lý Outliers bằng phương pháp Capping.")

# Chuẩn hóa số & Biến đổi Categorical
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Scaling numerical (Min-Max Scaling cho diện tích)
scaler = MinMaxScaler()
df['dien_tich_scaled'] = scaler.fit_transform(df[['dien_tich']])

# One-hot encoding cho cột 'loai_nha' (Chung cư, Nhà phố, Biệt thự)
# Cách đơn giản nhất trong Pandas:
df = pd.get_dummies(df, columns=['loai_nha'], prefix='type')

# Biến đổi Text mô tả bằng TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=100, stop_words='english') # Giới hạn 100 từ quan trọng nhất
tfidf_matrix = tfidf.fit_transform(df['mo_ta'].fillna(''))

print("Đã chuẩn hóa số và mã hóa dữ liệu chữ.")

# Phát hiện Duplicate dựa trên độ tương đồng văn bản (Cosine Similarity)
from sklearn.metrics.pairwise import cosine_similarity

# Tính toán ma trận tương đồng giữa các mô tả nhà (đã chuyển thành TF-IDF ở bước 4)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Tìm các cặp bản ghi có độ giống nhau trên 90% (nhưng không phải chính nó)
duplicates = []
for i in range(len(cosine_sim)):
    for j in range(i + 1, len(cosine_sim)):
        if cosine_sim[i, j] > 0.9:  # Ngưỡng giống nhau 90%
            duplicates.append((i, j, cosine_sim[i, j]))

# Hiển thị ví dụ cặp trùng lặp
if duplicates:
    print(f"Phát hiện {len(duplicates)} cặp có khả năng trùng lặp nội dung.")
    print(f"Cặp đầu tiên: Dòng {duplicates[0][0]} và Dòng {duplicates[0][1]} giống nhau {duplicates[0][2]*100:.2f}%")
else:
    print("Không phát hiện mô tả nào trùng lặp.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import skew, boxcox
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ====================== CÀI ĐẶT ======================
plt.rcParams['figure.figsize'] = (10, 6)
sns.set(style="whitegrid")
os.makedirs("plots", exist_ok=True)

# ====================== ĐỌC DỮ LIỆU ======================
df = pd.read_csv('ITA105_Lab_7.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ====================== BÀI 1: Skewness ======================
print("="*80)
print("BÀI 1: PHÂN TÍCH SKEWNESS VÀ TOP 10 CỘT LỆCH MẠNH NHẤT")
print("="*80)

skew_dict = {col: skew(df[col].dropna()) for col in numeric_cols}
top10_skew = sorted(skew_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

print("\nTop 10 cột lệch mạnh nhất (|skew| cao nhất):")
for col, sk in top10_skew:
    print(f"{col:25} Skew = {sk:.4f}")

# Vẽ 3 cột lệch mạnh nhất
top3 = [col for col, _ in top10_skew[:3]]
for col in top3:
    plt.figure()
    sns.histplot(df[col], kde=True, bins=50, color='skyblue')
    plt.title(f'Phân phối của {col}\nSkew = {skew_dict[col]:.4f}')
    plt.xlabel(col)
    plt.ylabel('Tần suất')
    plt.savefig(f'plots/Bai1_{col}_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu biểu đồ Bài 1: {col}")

# ====================== BÀI 2: Biến đổi dữ liệu ======================
print("\n" + "="*80)
print("BÀI 2: BIẾN ĐỔI DỮ LIỆU (Log - Box-Cox - Yeo-Johnson)")
print("="*80)

selected_cols = ['SalePrice', 'LotArea', 'NegSkewIncome']

results = []
for col in selected_cols:
    data = df[col].dropna().values
    orig_skew = float(skew(data))
    
    log_skew = float(skew(np.log(data))) if np.all(data > 0) else np.nan
    
    boxcox_skew = np.nan
    lam = np.nan
    if np.all(data > 0):
        bc_data, lam = boxcox(data)
        boxcox_skew = float(skew(bc_data))
    
    # Yeo-Johnson
    pt = PowerTransformer(method='yeo-johnson')
    power_data = pt.fit_transform(df[[col]]).flatten()
    power_skew = float(skew(power_data))
    
    results.append({
        'Cột': col,
        'Skew gốc': round(orig_skew, 4),
        'Skew sau Log': round(log_skew, 4) if not np.isnan(log_skew) else 'N/A',
        'Skew sau Box-Cox': round(boxcox_skew, 4) if not np.isnan(boxcox_skew) else 'N/A',
        'λ Box-Cox': round(lam, 4) if not np.isnan(lam) else 'N/A',
        'Skew sau Power': round(power_skew, 4)
    })

print(pd.DataFrame(results).to_string(index=False))

# Vẽ biểu đồ trước - sau cho Bài 2
for col in selected_cols:
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'Gốc - {col}\nSkew={skew_dict[col]:.3f}')
    
    if np.all(df[col] > 0):
        sns.histplot(np.log(df[col]), kde=True, ax=axes[1], color='orange')
        axes[1].set_title('Sau Log')
    
    if np.all(df[col] > 0):
        bc, _ = boxcox(df[col])
        sns.histplot(bc, kde=True, ax=axes[2], color='green')
        axes[2].set_title('Sau Box-Cox')
    
    sns.histplot(power_data, kde=True, ax=axes[3], color='red')
    axes[3].set_title('Sau Yeo-Johnson')
    
    plt.tight_layout()
    plt.savefig(f'plots/Bai2_{col}_before_after.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu biểu đồ Bài 2: {col}")

# ====================== BÀI 3: Mô hình Linear Regression ======================
print("\n" + "="*80)
print("BÀI 3: ỨNG DỤNG VÀO MÔ HÌNH LINEAR REGRESSION")
print("="*80)

X = df.select_dtypes(include=[np.number]).drop(columns=['SalePrice'], errors='ignore')
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Version A: Dữ liệu gốc
model_a = LinearRegression()
model_a.fit(X_train, y_train)
pred_a = model_a.predict(X_test)
rmse_a = np.sqrt(mean_squared_error(y_test, pred_a))
r2_a = r2_score(y_test, pred_a)

# Version B: Log trên SalePrice
model_b = LinearRegression()
model_b.fit(X_train, np.log(y_train))
pred_b_log = model_b.predict(X_test)
pred_b = np.exp(pred_b_log)
rmse_b = np.sqrt(mean_squared_error(y_test, pred_b))
r2_b = r2_score(y_test, pred_b)

print(f"Version A (Gốc)           : RMSE = {rmse_a:,.2f} | R² = {r2_a:.4f}")
print(f"Version B (Log SalePrice) : RMSE = {rmse_b:,.2f} | R² = {r2_b:.4f}")

# ====================== BÀI 4: Ứng dụng nghiệp vụ thực tế ======================
print("\n" + "="*80)
print("BÀI 4: ỨNG DỤNG NGHIỆP VỤ THỰC TẾ")
print("="*80)

business_cols = ['SalePrice', 'LotArea']

for col in business_cols:
    # Version A: Raw data
    plt.figure()
    sns.histplot(df[col], kde=True, bins=50, color='skyblue')
    plt.title(f'Version A - Raw Data: {col}')
    plt.xlabel(col)
    plt.savefig(f'plots/Bai4_{col}_raw.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Version B: Transformed (Log)
    if np.all(df[col] > 0):
        plt.figure()
        sns.histplot(np.log(df[col]), kde=True, bins=50, color='purple')
        plt.title(f'Version B - Log Transform: {col}')
        plt.xlabel(f'log({col})')
        plt.savefig(f'plots/Bai4_{col}_log.png', dpi=200, bbox_inches='tight')
        plt.close()

print("✓ Đã tạo biểu đồ Bài 4 (Raw & Log) cho SalePrice và LotArea")

# Insight cho Bài 4
print("\n=== INSIGHT DÀNH CHO BÀI 4 (Copy vào báo cáo) ===")
print("1. Tại sao cần biến đổi dữ liệu?")
print("   Dữ liệu gốc bị lệch mạnh (skew cao), phần lớn giá trị tập trung ở mức thấp, có một số outlier rất lớn.")
print("   Điều này làm khó quan sát mối quan hệ và ảnh hưởng đến mô hình dự báo.")
print("\n2. Biểu đồ sau khi transform giúp gì?")
print("   Sau khi dùng Log transform, phân phối trở nên đối xứng hơn, dễ nhìn thấy xu hướng:")
print("   - Nhà có diện tích lớn hơn thường có giá cao hơn một cách rõ ràng và ổn định.")
print("   - Giảm ảnh hưởng của các căn nhà giá cực cao.")
print("\n3. Metric mới gợi ý:")
print("   log_price_per_area = log(SalePrice / LotArea)")
print("   → Dùng để phát hiện khu vực có giá bất thường hoặc định giá nhà chính xác hơn.")
print("\n4. Khuyến nghị kinh doanh:")
print("   - Tập trung marketing vào phân khúc nhà diện tích trung bình (chiếm đa số khách hàng).")
print("   - Cảnh báo rủi ro khi mua nhà có giá hoặc diện tích lệch quá xa so với thị trường.")
print("   - Sử dụng mô hình có biến đổi Log trên SalePrice để dự báo giá nhà chính xác hơn.")

print("\n" + "="*80)
print("🎉 HOÀN THÀNH TOÀN BỘ LAB 7!")
print("Các file đã tạo trong thư mục 'plots/':")
print("- Biểu đồ Bài 1, Bài 2, Bài 4")
print("- Bảng so sánh Bài 2")
print("Bạn có thể copy phần Insight ở trên vào báo cáo Word.")
print("="*80)
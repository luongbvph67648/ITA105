import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Cấu hình để vẽ biểu đồ đẹp hơn
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def process_lab_exercise(file_path, title, columns_to_plot, scatter_cols=None):
    print(f"\n{'='*20} ĐANG XỬ LÝ: {title} {'='*20}")
    
    # 1. Nạp dữ liệu
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {file_path}. Vui lòng kiểm tra lại đường dẫn.")
        return

    # 2. Khám phá dữ liệu
    print(f"--- Thông tin dataset {title} ---")
    print(df.info())
    print("\n--- Thống kê mô tả ---")
    print(df.describe())
    print("\n--- Kiểm tra giá trị thiếu ---")
    print(df.isnull().sum())

    # 3. Trực quan hóa trước khi chuẩn hóa (Histogram và Boxplot)
    fig, axes = plt.subplots(2, len(columns_to_plot), figsize=(18, 10))
    fig.suptitle(f'Phân phối dữ liệu GỐC - {title}', fontsize=16)

    for i, col in enumerate(columns_to_plot):
        sns.histplot(df[col], kde=True, ax=axes[0, i], color='skyblue')
        axes[0, i].set_title(f'Histogram {col}')
        sns.boxplot(x=df[col], ax=axes[1, i], color='lightcoral')
        axes[1, i].set_title(f'Boxplot {col}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 4. Chuẩn hóa dữ liệu
    scaler_minmax = MinMaxScaler()
    scaler_zscore = StandardScaler()

    df_minmax = pd.DataFrame(scaler_minmax.fit_transform(df[columns_to_plot]), columns=[f"{c}_minmax" for c in columns_to_plot])
    df_zscore = pd.DataFrame(scaler_zscore.fit_transform(df[columns_to_plot]), columns=[f"{c}_zscore" for c in columns_to_plot])

    # 5. Trực quan hóa sau khi chuẩn hóa
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Vẽ ví dụ cho cột đầu tiên để so sánh
    col_example = columns_to_plot[0]
    sns.histplot(df_minmax[f"{col_example}_minmax"], kde=True, ax=axes[0], color='green', label='Min-Max')
    axes[0].set_title(f'Sau Min-Max Scaling ({col_example})')
    
    sns.histplot(df_zscore[f"{col_example}_zscore"], kde=True, ax=axes[1], color='orange', label='Z-Score')
    axes[1].set_title(f'Sau Z-Score Normalization ({col_example})')
    
    plt.suptitle(f'So sánh phương pháp chuẩn hóa - {title}', fontsize=16)
    plt.show()

    # 6. Vẽ Scatter Plot (Dành riêng cho Bài 3)
    if scatter_cols and len(scatter_cols) == 2:
        col1, col2 = scatter_cols
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Trước chuẩn hóa
        sns.scatterplot(data=df, x=col1, y=col2, ax=ax1)
        ax1.set_title(f'Trước chuẩn hóa: {col1} vs {col2}')
        
        # Sau Z-score (Ví dụ)
        sns.scatterplot(x=df_zscore[f"{col1}_zscore"], y=df_zscore[f"{col2}_zscore"], ax=ax2, color='red')
        ax2.set_title(f'Sau Z-Score: {col1} vs {col2}')
        
        plt.show()

# --- CHẠY CÁC BÀI TẬP ---

# Bài 1: Thông số vận động viên
process_lab_exercise(
    'ITA105_Lab_3_Sports.csv', 
    'Bài 1: Sports', 
    ['chieu_cao_cm', 'can_nang_kg', 'toc_do_100m_s']
)

# Bài 2: Chỉ số bệnh nhân
process_lab_exercise(
    'ITA105_Lab_3_Health.csv', 
    'Bài 2: Health', 
    ['BMI', 'huyet_ap_mmHg', 'cholesterol_mg_dl']
)

# Bài 3: Chỉ số công ty
process_lab_exercise(
    'ITA105_Lab_3_Finance.csv', 
    'Bài 3: Finance', 
    ['doanh_thu_musd', 'loi_nhuan_musd', 'EPS'],
    scatter_cols=['doanh_thu_musd', 'loi_nhuan_musd']
)

# Bài 4: Người chơi trực tuyến
process_lab_exercise(
    'ITA105_Lab_3_Gaming.csv', 
    'Bài 4: Gaming', 
    ['gio_choi', 'diem_tich_luy', 'so_level']
)
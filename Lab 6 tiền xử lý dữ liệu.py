import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def normalize_image(image):
    """Chuẩn hóa pixel từ [0-255] về [0-1]."""
    return image.astype(np.float32) / 255.0

def apply_brightness(image, factor):
    """Thay đổi độ sáng của ảnh."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def rotate_image(image, angle):
    """Xoay ảnh một góc cụ thể."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def add_gaussian_noise(image, mean=0, sigma=0.1):
    """Thêm nhiễu Gaussian vào ảnh (giả định ảnh đã normalize [0, 1])."""
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 1)

def random_crop_zoom(image, scale_range=(0.7, 0.9)):
    """Thực hiện Random Crop và Zoom về kích thước gốc."""
    h, w = image.shape[:2]
    scale = random.uniform(*scale_range)
    new_h, new_w = int(h * scale), int(w * scale)
    
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    
    cropped = image[top:top+new_h, left:left+new_w]
    return cv2.resize(cropped, (w, h))

# --- BÀI 1: CĂN HỘ / MẶT TIỀN ---
def process_task_1(images_list):
    print("Đang xử lý Bài 1...")
    results_orig = []
    results_aug = []
    
    for img in images_list[:5]:
        # 1. Resize
        img_resized = cv2.resize(img, (224, 224))
        results_orig.append(img_resized)
        
        # 2. Augmentation
        aug = cv2.flip(img_resized, 1) # Horizontal flip
        aug = rotate_image(aug, random.uniform(-15, 15)) # Rotate ±15
        aug = apply_brightness(aug, random.uniform(0.8, 1.2)) # Brightness ±20%
        
        # 3. Chuyển Grayscale & Chuẩn hóa
        gray = cv2.cvtColor(aug, cv2.COLOR_RGB2GRAY)
        norm = normalize_image(gray)
        results_aug.append(norm)

    # Hiển thị
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        axes[0, i].imshow(results_orig[i])
        axes[0, i].set_title(f"Gốc {i+1}")
        axes[1, i].imshow(results_aug[i], cmap='gray')
        axes[1, i].set_title(f"Aug {i+1}")
    plt.tight_layout()
    plt.show()

# --- BÀI 2: XE Ô TÔ / XE MÁY ---
def process_task_2(img):
    print("Đang xử lý Bài 2...")
    # 1. Resize
    img = cv2.resize(img, (224, 224))
    
    # 2. Augmentation & Grayscale (Tùy chọn)
    img_aug = apply_brightness(img, random.uniform(0.85, 1.15)) # ±15%
    img_aug = rotate_image(img_aug, random.uniform(-10, 10)) # ±10
    
    # 3. Chuẩn hóa & Thêm Gaussian Noise
    norm = normalize_image(img_aug)
    noisy = add_gaussian_noise(norm, sigma=0.05)
    
    return noisy

# --- BÀI 3: TRÁI CÂY / NÔNG SẢN ---
def process_task_3(img):
    print("Đang xử lý Bài 3...")
    # Resize & Normalize
    img = cv2.resize(img, (224, 224))
    
    augmented_images = []
    for _ in range(9):
        # Augmentation pipeline
        aug = img.copy()
        if random.random() > 0.5: aug = cv2.flip(aug, 1) # Flip
        aug = rotate_image(aug, random.uniform(-30, 30)) # Rotation
        aug = random_crop_zoom(aug) # Random Crop & Zoom
        
        augmented_images.append(normalize_image(aug))

    # Hiển thị grid 3x3
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(augmented_images[i])
        ax.axis('off')
    plt.suptitle("Bài 3: Grid 3x3 Augmentation")
    plt.show()

# --- BÀI 4: PHÒNG / NỘI THẤT ---
def process_task_4(img):
    print("Đang xử lý Bài 4...")
    img_orig = cv2.resize(img, (224, 224))
    
    aug_results = []
    for _ in range(3):
        aug = rotate_image(img_orig, random.uniform(-15, 15))
        aug = cv2.flip(aug, 1)
        aug = apply_brightness(aug, random.uniform(0.8, 1.2))
        
        # Grayscale & Normalize
        gray = cv2.cvtColor(aug, cv2.COLOR_RGB2GRAY)
        aug_results.append(normalize_image(gray))

    # Hiển thị
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_orig)
    axes[0].set_title("Ảnh gốc")
    for i in range(3):
        axes[i+1].imshow(aug_results[i], cmap='gray')
        axes[i+1].set_title(f"Augmented {i+1}")
    plt.show()

# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    # Tạo ảnh mẫu (dummy data) nếu bạn chưa có file ảnh thật
    # Trong thực tế, bạn hãy dùng cv2.imread('path/to/image.jpg')
    dummy_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    # Vẽ một vài hình khối vào ảnh mẫu để dễ quan sát sự thay đổi
    cv2.rectangle(dummy_img, (50, 50), (200, 200), (255, 0, 0), -1)
    cv2.circle(dummy_img, (150, 150), 50, (0, 255, 0), -1)
    
    # Giả lập danh sách 5 ảnh cho Bài 1
    sample_images = [dummy_img.copy() for _ in range(5)]
    
    # Thực thi các bài tập
    process_task_1(sample_images)
    
    processed_task_2 = process_task_2(dummy_img)
    print("Bài 2 hoàn thành. Kích thước:", processed_task_2.shape)
    
    process_task_3(dummy_img)
    
    process_task_4(dummy_img)


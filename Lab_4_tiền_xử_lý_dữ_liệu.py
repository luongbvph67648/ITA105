import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Dữ liệu
data = {"hours": [1,2,3,4,5,6,7,8],
        "score": [2,4,5,6,7,8,8.5,9]}
df = pd.DataFrame(data)

# Tách dữ liệu
x = df[["hours"]]
y = df[["score"]]

# Huấn luyện
model = LinearRegression()
model.fit(x, y)

# Dự đoán
new_hours = [[6], [9]]
preds = model.predict(new_hours)
print("Dự đoán:", preds)

# Vẽ biểu đồ
plt.scatter(x, y, label="Dữ liệu thực tế")
plt.plot(x, model.predict(x), color='red', label="Đường hồi quy")
plt.xlabel("Số giờ học")
plt.ylabel("Điểm thi")
plt.legend()
plt.show()

# Đánh giá
y_pred = model.predict(x)
print("R2 score:", r2_score(y, y_pred))
print("MAE:", mean_absolute_error(y, y_pred))
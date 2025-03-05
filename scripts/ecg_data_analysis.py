import pandas as pd
import numpy as np
import random
import os
from plotly.offline import plot
import plotly.graph_objs as go
from collections import Counter

# Đảm bảo lưu file vào thư mục `ECG_Heartbeat_Classification/images/` đúng vị trí
base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))  # Lấy thư mục cha của `scripts`
image_dir = os.path.join(base_dir, "images")  # Đặt thư mục lưu file vào `images/`
os.makedirs(image_dir, exist_ok=True)  # Tạo thư mục nếu chưa có

df = pd.read_csv("../input/mitbih_test.csv", header=None)
print(df.shape)
print(Counter(df[187].values))

Y = np.array(df[187].values).astype(np.int8)
X = np.array(df[list(range(187))].values)

indexes = random.sample(list(range(df.shape[0])), 10)

for i in indexes:
    data = [go.Scatter(
              x=list(range(187)),
              y=X[i, :])]
    
    plot({"data": data,
          "layout": {"title": "Heartbeat Class : %s " % Y[i]}}, filename=os.path.join(image_dir, f"{i}.html"))

import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
df = pd.read_excel("results.xlsx", sheet_name=6)

# 获取X和Y列的数据
x = df["权重"]
y = df["cov_mean↑"]


# 绘制折线图
plt.plot(x, y, marker='o', linestyle='-')
plt.xlabel("w_dg")
plt.ylabel("cov_mean")
plt.title("title")

# 显示图形
# plt.show()
plt.savefig("ttest.tiff", format="tiff")

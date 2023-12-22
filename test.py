import torch

# 假设你有一个大小为 [4, 1] 的张量
tensor_variable = torch.tensor([[9.8], [10.1], [8.9], [11.0], [10.2]])

# 判断是否有接近于10的数字
threshold = 0.5  # 设置阈值，表示接近于10的范围
indices_near_10 = torch.nonzero(torch.abs(tensor_variable - 10) < threshold)

# 打印结果
if indices_near_10.numel() > 0:
    print("张量中包含接近于10的数字，下标为:", indices_near_10)
else:
    print("张量中没有接近于10的数字")



import torch
import pickle
with open(r"D:\pycharmProjects\DGDM\DGDM\data\qm9_processed\discriminator_test_data.pkl", 'rb') as file:
    # 使用pickle.load()加载数据
    data = pickle.load(file)

print(data[0])
print(data[len(data) // 2])

# a = torch.ones(10)
# b = torch.zeros(10)
# c = torch.nn.Sigmoid()
# res = c(a)
# print(a)
# print(b)
# print(res)
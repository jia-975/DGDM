import os
import pickle

load_path = os.path.join("data/qm9_processed")
with open(os.path.join(load_path, "test_data_200.pkl"), "rb") as fin:
    test_data = pickle.load(fin)


def prn_obj(obj):
  print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]) )


print(len(test_data))
for i in range(100, 110):
    print(test_data[i])
    print()
    prn_obj(test_data[i])
    print(i)


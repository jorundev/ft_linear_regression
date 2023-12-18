from common import read_normalized_model, read_csv, normalize_dataset
from typing import Tuple

a, b = read_normalized_model()
_, _, feature_data, target_data = read_csv("data.csv")

feature_data, target_data, _, _, _, _ = normalize_dataset(feature_data, target_data)

m = len(feature_data)

def predict_target(feature: float):
	return a * feature + b

def cost():
	ret = 0
	for i in range(0, m):
		ret += (predict_target(feature_data[i]) - target_data[i])**2
	return ret / (2 * m)

print("Model cost:", cost())

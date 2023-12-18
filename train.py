from common import read_csv, normalize_dataset
from typing import Tuple

feature_name, target_name, feature_data, target_data = read_csv("data.csv")
learning_rate = 0.01

feature_data, target_data, min_feature_original, max_feature_original, min_target_original, max_target_original = normalize_dataset(feature_data, target_data)

def denormalize_model(model: Tuple[float, float]) -> Tuple[float, float]:
    slope, intercept = model

    slope_original = slope * (max_target_original - min_target_original) / (max_feature_original - min_feature_original)
    intercept_original = intercept * (max_target_original - min_target_original) - slope_original * min_feature_original + min_target_original
    return slope_original, intercept_original

def predict_target(feature: float, model: Tuple[float, float]):
	return model[0] * feature + model[1]

def iterate_training(m: int, model: Tuple[float, float]) -> Tuple[float, float]:
	b_correction = 0
	for i in range(0, m):
		b_correction += predict_target(feature_data[i], model) - target_data[i]
	b_correction *= learning_rate / m

	a_correction = 0
	for i in range(0, m):
		a_correction += (predict_target(feature_data[i], model) - target_data[i]) * feature_data[i]
	a_correction *= learning_rate / m

	return model[0] - a_correction, model[1] - b_correction

model = 0, 0
m = len(feature_data)
for i in range(0, 10000):
	model = iterate_training(m, model)

normalized_model_file = open("nmodel", "w")
normalized_model_file.write(str(model[0]) + "\n" + str(model[1]))

denormalized = denormalize_model(model)

model_file = open("model", "w")
model_file.write(str(denormalized[0]) + "\n" + str(denormalized[1]))

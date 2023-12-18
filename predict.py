from common import read_csv, read_model;
from typing import Tuple

def predict_target(feature: float, model: Tuple[float, float]):
	return model[0] * feature + model[1]

a, b = read_model()

feature, target, _, _ = read_csv("data.csv")

while True:
	try:
		value = float(input(feature + ": "))
		break
	except:
		print("Incorrect input. Try again")

print("predicted " + target + ": " + str(predict_target(value, (a, b))))

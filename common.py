import csv
from typing import Tuple, List

def read_csv(path: str) -> Tuple[str, str, list, list]:
	try:
		csv_data = open(path, "r").read()
	except:
		print("Could not read", path)
		exit(1)

	lines = csv_data.splitlines()

	dataset_len = len(lines)
	if dataset_len < 3:
		print("Invalid dataset")
		exit(1)

	csv_reader = csv.reader(lines)

	names = next(csv_reader)
	feature = names[0]
	target = names[1]

	feature_data = []
	target_data = []

	for row in csv_reader:
		try:
			feature_data.append(float(row[0]))
			target_data.append(float(row[1]))
		except:
			print("Expected numbers in CSV")
			exit(1)

	return feature, target, feature_data, target_data

def read_model_file(path: str):
	try:
		model_lines = open(path, "r").read().splitlines()
		return float(model_lines[0]), float(model_lines[1])
	except:
		print("Error while reading model")
		exit(1)

def read_model() -> Tuple[float, float]:
	return read_model_file("model")

def read_normalized_model() -> Tuple[float, float]:
	return read_model_file("nmodel")

def normalize_dataset(feature_data: List[float], target_data: List[float]) -> Tuple[List[float], List[float], float, float, float, float]:
	max_feature, min_feature = max(feature_data), min(feature_data)
	max_target, min_target = max(target_data), min(target_data)

	if max_feature == min_feature or max_target == min_target:
		print("Invalid dataset")
		exit(1)

	feature_data = [(x - min_feature) / (max_feature - min_feature) for x in feature_data]
	target_data = [(x - min_target) / (max_target - min_target) for x in target_data]

	return feature_data, target_data, min_feature, max_feature, min_target, max_target

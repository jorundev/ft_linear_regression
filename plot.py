import matplotlib.pyplot as plt
from common import read_csv, read_model

(feature, target, feature_data, target_data) = read_csv("data.csv")
a, b = read_model()

plt.scatter(feature_data, target_data, label='Dataset', color='blue')

min_feature, max_feature = min(feature_data), max(feature_data)
min_target, max_target = min(target_data), max(target_data)

plt.xlim(min_feature, max_feature)
plt.ylim(min_target, max_target)

plt.xlabel(feature)
plt.ylabel(target)
plt.title(target + ' by ' + feature)

plt.axline(xy1=(0, b), slope=a, color='red', label='Model')

plt.legend()
plt.savefig('plot.png')

import numpy as np
import matplotlib.pyplot as plt

# Load both scenario training sets
# X1 = np.load("exports/concat_scenario1_train/X_train.npz")['x']
# y1 = np.load("exports/concat_scenario1_train/y_train.npz")['y']
# X2 = np.load("exports/concat_scenario2_train/X_train.npz")['x']
# y2 = np.load("exports/concat_scenario2_train/y_train.npz")['y']

X_all = np.load("exports/X_train_s05.npz")['x']
y_all = np.load("exports/y_train_s05.npz")['y']

# Concatenate
# X_all = np.concatenate([X1, X2], axis=0)
# y_all = np.concatenate([y1, y2], axis=0)

print("Combined X shape:", X_all.shape)
print("Combined y shape:", y_all.shape)
print("Number of samples per class:", {int(lbl): sum(y_all==lbl) for lbl in np.unique(y_all)})
print("Number of channels:", X_all.shape[1])

# Plot class distribution
plt.figure(figsize=(6,4))
plt.hist(y_all, bins=np.arange(y_all.min(), y_all.max()+2)-0.5, rwidth=0.8)
plt.xlabel("Class label")
plt.ylabel("Number of samples")
plt.title("Class distribution in combined training set")
plt.xticks(sorted(np.unique(y_all)))
plt.show()

# Plot a few random samples' signals
plt.figure(figsize=(12,4))
for i in range(3):
    plt.plot(X_all[np.random.randint(len(X_all)), :], alpha=0.7, label=f'Random sample {i+1}')
plt.title("Random training samples (all channels)")
plt.xlabel("Channel")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
#question do ou want to export data Y/n
prompt = input("Do you want to export data? (yes/no)")
if   prompt.lower() == 'yes':
    np.savez_compressed("exports/general_training_dataset_s05.npz", x=X_all, y=y_all)
    print("Exported general training dataset to exports/general_training_dataset_s05.npz")
else:
    print("Data not exported")

# #load
# data = np.load("exports/general_training_dataset.npz")
# X = data['x']
# y = data['y']
# print("Loaded general training dataset from exports/general_training_dataset.npz")
# print("X shape:", X.shape)
# print("y shape:", y.shape)
# print("Number of samples per class:", {int(lbl): sum(y==lbl) for lbl in np.unique(y)})
# print("Number of channels:", X.shape[1])

# # Plot class distribution
# plt.figure(figsize=(6,4))
# plt.hist(y, bins=np.arange(y.min(), y.max()+2)-0.5, rwidth=0.8)
# plt.xlabel("Class label")
# plt.ylabel("Number of samples")
# plt.title("Class distribution in loaded training set")
# plt.xticks(sorted(np.unique(y)))
# plt.show()

# # Plot a few random samples' signals
# plt.figure(figsize=(12,4))
# for i in range(3):
#     plt.plot(X[np.random.randint(len(X)), :], alpha=0.7, label=f'Random sample {i+1}')
# plt.title("Random training samples (all channels)")
# plt.xlabel("Channel")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.show()

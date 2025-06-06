!pip install wfdb
# Import necessary libraries
import os
import numpy as np
import pandas as pd
import wfdb

# Path to dataset (adjust as needed)
dataset_path = '/content/ECG_dataset/ECG_dataset/'

# Clean up record names (remove any suffixes like - or _)
records = [f.split('.')[0] for f in os.listdir(dataset_path) if f.endswith('.atr')]
records = list(set(records))
records.sort()

# Identify unique labels in the dataset
unique_labels = set()
for record_name in records:
    try:
        annotations = wfdb.rdann(os.path.join(dataset_path, record_name), 'atr')
        unique_labels.update(annotations.symbol)
    except Exception as e:
        print(f"Exception occurred while reading {record_name}: {e}")

print(f"Unique labels: {unique_labels}")

# Window size setup
window_size = 187
half_window = window_size // 2

# Mapping of heartbeat types to labels
mapping = {
    'N': 0, 'L': 0, 'R': 0, 'B': 0,   # Normal beats
    'A': 1, 'a': 1, 'J': 1, 'S': 1,   # Supraventricular beats
    'V': 2, 'E': 2,                   # Ventricular beats
    'F': 3,                           # Fusion beats
    '/': 4, 'f': 4, 'Q': 4, 'j': 4    # Unknown beats
}

# Initialize lists to store extracted data
heartbeats = []
labels = []
skip = 0
invalid = 0

# Read ECG signals and annotations
for record_num in records:
    set_path = os.path.join(dataset_path, record_num)

    try:
        # Load ECG record and annotations
        record = wfdb.rdrecord(set_path)
        ann = wfdb.rdann(set_path, "atr")

        ecg_signal = record.p_signal[:, 0]  # Take lead 1 (or modify for other leads)
        r_peaks = ann.sample
        label = ann.symbol

        for i, n in enumerate(r_peaks):
            start = n - half_window
            end = start + window_size

            # Skip if the signal is too close to the start or end
            if start < 10 or end > len(ecg_signal) - 10:
                skip += 1
                continue

            heart_beat = ecg_signal[start:end]
            l = mapping.get(label[i], -1)

            if l != -1:
                heartbeats.append(heart_beat)
                labels.append(l)
            else:
                invalid += 1

    except Exception as e:
        print(f"Exception occurred at {record_num}: {e}")

# Create a DataFrame from the extracted data
df = pd.DataFrame(heartbeats)
df['label'] = labels

# Save the DataFrame to a CSV file
csv_path = '/content/mitbih_ecg_processed.csv'
df.to_csv(csv_path, index=False)

print(f"The data has been stored successfully to {csv_path}")
print(f"Total heartbeats: {len(heartbeats)}, Skipped: {skip}, Invalid: {invalid}")

df=df.sample(frac=1,random_state=42).reset_index(drop=True)
df.head()
print(df.isnull().sum().sum())  # Total count of missing values in the DataFrame
from sklearn.model_selection import train_test_split

X=df.iloc[:,:-1]#all except last column
y=df.iloc[:,-1]#only last column

#split data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print(f"Data is split!\nTraning data:{X_train.shape}....{y_train.shape}")
print(f"Testing data:{X_test.shape}....{y_test.shape}")

#normalize
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#use fit_transform only on train data to ensure consistency
#fit() calculates mean and std. We want to use that same value in case of the test data as well
print("Data is normalized using minmax!")
print(f"Traning data:{X_train.shape}....{y_train.shape}")
print(f"Testing data:{X_test.shape}....{y_test.shape}")

training=pd.DataFrame(X_train)
training['label']=pd.Series(y_train).reset_index(drop=True)
training.to_csv("Training_normalized.csv",index=False)
print("Training data saved")
training=pd.DataFrame(X_test)
training['label']=pd.Series(y_test).reset_index(drop=True)
training.to_csv("Testing_normalized.csv",index=False)
print("Testing data saved")


#plot PCA variance curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Fit PCA to the training data
pca = PCA(n_components=186)
pca.fit(X_train)

# Calculate cumulative variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative variance
plt.figure(figsize=(8,5))
plt.plot(range(1, 187), cumulative_variance, marker='o', linestyle='--')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA: Explained Variance vs. Number of Components")
plt.axhline(y=0.95, color='r', linestyle='--')  # Mark 95% variance
plt.grid()
plt.show()

#perform PCA with 30 components
from sklearn.decomposition import PCA
pca=PCA(n_components=30)
train_pca=pca.fit_transform(X_train)
test_pca=pca.transform(X_test)

print("Data dimensionaility reduced with PCA!")
print(f"Traning data:{train_pca.shape}....{y_train.shape}")
print(f"Testing data:{test_pca.shape}....{y_test.shape}")
training=pd.DataFrame(train_pca)
training['label']=pd.Series(y_train).reset_index(drop=True)
training.to_csv("Training_PCA.csv",index=False)
print("Training data saved")
testing=pd.DataFrame(test_pca)
testing['label']=pd.Series(y_test).reset_index(drop=True)
testing.to_csv("Testing_PCA.csv",index=False)
print("Testing data saved")

print(set(y_train) - set(y_test))  # Check if train labels contain extra classes
print(training.duplicated().sum())  # Count duplicate rows in training set
print(testing.duplicated().sum())   # Count duplicate rows in test set


#plot histogram fro data
import matplotlib.pyplot as plt

def plot_label_distribution(y, title):
    # Count label occurrences
    count = pd.Series(y).value_counts()

    # Ensure all classes (0 to 4) are present
    classes = [0, 1, 2, 3, 4]
    count = count.reindex(classes, fill_value=0)

    # Plot bar chart
    plt.bar(count.index, count.values, color='lightblue', edgecolor='black')

    # Add text labels on top of bars
    for i, v in enumerate(count.values):
        plt.text(i, v + 5, str(v), ha='center')

    # Formatting
    plt.xlabel("Class Labels")
    plt.ylabel("Samples")
    plt.xticks(rotation=0)
    plt.title(title)

# Create subplots for Training and Test labels
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_label_distribution(y_train, "Training Labels Distribution")

plt.subplot(1, 2, 2)
plot_label_distribution(y_test, "Test Labels Distribution")

plt.tight_layout()
plt.show()

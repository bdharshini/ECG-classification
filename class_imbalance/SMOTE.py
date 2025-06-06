#SMOTE data preparation
#Shld be done before PCA
import pandas as pd
train_data = pd.read_csv("Training_normalized.csv")
test_data = pd.read_csv("Testing_normalized.csv")

# Split features and labels
X_train = train_data.iloc[:, :-1]  # All columns except the last one
y_train = train_data.iloc[:, -1]   # Last column as target
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]
print(f"Train:{X_train.shape}")
print(f"Train:{X_test.shape}")

from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
# Step 1: Apply SMOTE to training data only (important!)
smote = SMOTE(random_state=42)
X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)

print("SMOTE oversampling done!")
print(f"Oversampled training data shape: {X_train_oversampled.shape}....{y_train_oversampled.shape}")

# Step 2: Now apply PCA to the oversampled data
pca = PCA(n_components=30)
train_pca = pca.fit_transform(X_train_oversampled)
test_pca = pca.transform(X_test)

print("Data dimensionality reduced with PCA!")
print(f"Training data: {train_pca.shape}....{y_train_oversampled.shape}")
print(f"Testing data: {test_pca.shape}....{y_test.shape}")

training=pd.DataFrame(train_pca)
training['label']=pd.Series(y_train_oversampled).reset_index(drop=True)
training.to_csv("Training_PCA_oversampled.csv",index=False)
print("Training data saved")
testing=pd.DataFrame(test_pca)
testing['label']=pd.Series(y_test).reset_index(drop=True)
testing.to_csv("Testing_PCA_oversampled.csv",index=False)
print("Testing data saved")
#histogram representation of SMOTE
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
plot_label_distribution(y_train_oversampled, "Training Labels Distribution")

plt.tight_layout()
plt.show()

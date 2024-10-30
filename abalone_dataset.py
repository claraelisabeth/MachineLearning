import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler

# Load the Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = ["Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
abalone_df = pd.read_csv(url, header=None, names=column_names)

# Display the first few rows
print(abalone_df.head())

# Basic Characteristics
num_samples = abalone_df.shape[0]
num_attributes = abalone_df.shape[1]

# Attribute Types
attribute_types = {
    'Sex': 'Nominal (Categorical)',
    'Length': 'Continuous (Interval)',
    'Diameter': 'Continuous (Interval)',
    'Height': 'Continuous (Interval)',
    'Whole_weight': 'Continuous (Interval)',
    'Shucked_weight': 'Continuous (Interval)',
    'Viscera_weight': 'Continuous (Interval)',
    'Shell_weight': 'Continuous (Interval)',
    'Rings': 'Integer (can be treated as Ordinal)'
}

# Print Characteristics
print(f"Number of Samples: {num_samples}")
print(f"Number of Attributes: {num_attributes}")
print("Attribute Types:")
for attr, tipo in attribute_types.items():
    print(f"- {attr}: {tipo}")

# Distribution of Input Attributes
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16))
axes = axes.flatten()

for i, column in enumerate(abalone_df.columns[1:8]):  # Skip 'Sex' and 'Rings'
    sns.histplot(abalone_df[column], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {column}')
    
plt.tight_layout()
plt.show()

# Distribution of Target Attribute (Rings)
plt.figure(figsize=(10, 6))
sns.histplot(abalone_df['Rings'], bins=30, kde=True)
plt.title('Distribution of Rings (Target Attribute)')
plt.xlabel('Rings')
plt.ylabel('Frequency')
plt.show()

# Handling Categorical Data - One Hot Encoding
#abalone_encoded = pd.get_dummies(abalone_df, columns=['Sex'], drop_first=True)

# Splitting the Data into Features and Target
#X = abalone_encoded.drop('Rings', axis=1)
#y = abalone_encoded['Rings']

# Train-test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the Numeric Features
#scaler = StandardScaler()
#X_train.iloc[:, :-2] = scaler.fit_transform(X_train.iloc[:, :-2])
#X_test.iloc[:, :-2] = scaler.transform(X_test.iloc[:, :-2])

# Display shapes of train and test sets
#print(f"Training set shape: {X_train.shape}")
#print(f"Test set shape: {X_test.shape}")
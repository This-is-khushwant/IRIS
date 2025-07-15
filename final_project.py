# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load your dataset
# df = pd.read_csv('IRIS.csv')

# Sample data if you're using the inbuilt iris dataset:
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['Species'] = iris.target
species_map = dict(zip(range(3), iris.target_names))
df['Species'] = df['Species'].map(species_map)

# Step 3: View the data
print(df.head())

# Step 4: Split into features and target
X = df.iloc[:, :-1]  # all columns except 'Species'
y = df['Species']

# Step 5: Encode class labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 7: Choose a classifier (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Predict and evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))


# Step 9: Predict on new data
# sepal_length,sepal_width,petal_length,petal_width
sample = [[5.1, 3.5, 1.4, 0.2]] 
prediction = model.predict(sample)
predicted_class = le.inverse_transform(prediction)
print("Predicted Species:", predicted_class[0])


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load Data
df = pd.read_csv("accident_data.csv")
print(df.head())

# Step 2: Data Preprocessing
df.dropna(inplace=True)

# Encoding categorical features
df['Weather_Condition'] = df['Weather_Condition'].astype('category').cat.codes
df['Road_Condition'] = df['Road_Condition'].astype('category').cat.codes

# Step 3: Feature Selection
features = df.drop(['Accident_Severity'], axis=1)
labels = df['Accident_Severity']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Step 5: Model Building
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluation
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Step 7: Visualization
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

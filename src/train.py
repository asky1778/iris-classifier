import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# create outputs folder if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# load dataset
iris = load_iris()
X, y = iris.data, iris.target

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# accuracy
accuracy = model.score(X_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# classification matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, 
yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.savefig('outputs/confusion_matrix.png')
plt.close()

# save model
joblib.dump(model, "outputs/model.joblib")

print("model and confusion matrix saved in outputs/")




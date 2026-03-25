from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
x = iris.data
y = iris.target

# split data
x_train, x_test, y_train, y_test =train_test_split(x, y, test_size = 0.2, random_state = 42 )

# Train model 
model = LogisticRegression(max_iter = 200)
model.fit(x_train, y_train)

# prediction
y_pred = model.predict(x_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy ",accuracy)

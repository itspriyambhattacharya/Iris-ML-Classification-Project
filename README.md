# Iris Flower Classification using Machine Learning (KNN Classifier)

This project demonstrates a complete end-to-end Machine Learning
workflow using the **Iris Flower Dataset**, one of the most widely used
benchmark datasets in pattern recognition.
The objective of this project is to classify iris flowers into three
species --- **Setosa**, **Versicolor**, and **Virginica** --- based on
sepal and petal measurements.

The project includes: - Loading the dataset

- Exploratory data visualization
- Stratified train-test split
- Feature scaling
- Model training using K-Nearest Neighbors (KNN)
- Model evaluation (accuracy, classification report, confusion matrix)
- Visualization of class distribution and confusion matrix

## 📁 Dataset Description

The Iris dataset includes **150 samples**, belonging to three species:

Feature Description

---

Sepal Length in cm
Sepal Width in cm
Petal Length in cm
Petal Width in cm
Target Class Setosa, Versicolor, Virginica

## 📌 Project Workflow

### **1. Import Required Libraries**

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
```

### **2. Load the Iris Dataset**

```python
data = load_iris(as_frame=True, return_X_y=False)
df = data.frame
X, y = load_iris(return_X_y=True, as_frame=False)
```

### **3. Stratified Train-Test Split**

```python
sp = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for trainIdx, testIdx in sp.split(X, y):
    X_train, X_test = X[trainIdx], X[testIdx]
    y_train, y_test = y[trainIdx], y[testIdx]
```

### **4. Feature Scaling**

```python
scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)
```

### **5. Model Training**

```python
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
```

### **6. Predictions and Evaluation**

```python
y_pred = knn.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy is: {acc}")

print("Classification Report: ")
cr = classification_report(y_test, y_pred)
print(cr)

print("Confusion Matrix: ")
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

## 📊 Visualizations

### **Count Plot**

```python
sns.countplot(x=y)
plt.title("Types of Flowers")
plt.xticks([0, 1, 2], ["Setosa", "Versicolor", "Virginica"])
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
```

### **Confusion Matrix Heatmap**

```python
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='cool')
plt.title('Confusion Matrix')
plt.xlabel('Predited')
plt.ylabel('Actual')
plt.show()
```

## ▶️ How to Run the Project

### Clone the Repository

```bash
git clone https://github.com/itspriyambhattacharya/iris-ml-classification-project.git
cd iris-ml-classification-project
```

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 📌 Future Improvements

- Add more classifiers (SVM, Decision Tree, Random Forest)
- Hyperparameter tuning using GridSearchCV
- Apply PCA
- Deploy using Streamlit or Flask

## 📜 License

MIT License

## 🙌 Acknowledgements

- UCI ML Repository

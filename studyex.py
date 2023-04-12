from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 데이터 로드
iris = load_iris()

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

# 의사결정트리 모델 생성
clf = DecisionTreeClassifier()

# 모델 학습
clf.fit(X_train, y_train)

# 테스트 데이터로 예측
y_pred = clf.predict(X_test)

# 정확도 측정
acc = accuracy_score(y_test, y_pred)

# 새로운 데이터로 학습
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])
new_target = np.array([0])

clf.fit(new_data, new_target)

# 새로운 데이터로 예측
new_pred = clf.predict(new_data)

print("Accuracy:", acc)
print("New prediction:", new_pred)

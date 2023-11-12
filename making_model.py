import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# CSV 파일에서 데이터 불러오기
df = pd.read_csv('it_job_skills_extended.csv')

# Category 열을 타겟 변수로, Skills 열을 텍스트 데이터로 사용
X = df['Skills']
y = df['Category']

# 텍스트 데이터를 TF-IDF 벡터로 변환
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(X)

# 학습 및 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 다중 분류를 위한 Naive Bayes 모델 초기화 및 학습
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# 모델 예측
y_pred = classifier.predict(X_test)

# 정확도 및 분류 보고서 출력
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print(report)

# 분류 보고서 시각화
report_df = pd.DataFrame(report).transpose()

# 정밀도, 재현율, F1 점수 시각화
plt.figure(figsize=(10, 5))
sns.heatmap(report_df[['precision', 'recall', 'f1-score']], annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Classification Report')
plt.show()

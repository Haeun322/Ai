import joblib
import matplotlib.pyplot as plt

# 모델 로드
loaded_classifier = joblib.load("model2.pkl")

category_length = int(input("Category length: "))
user_skills = []
user_skill_donate = [0, 0, 0, 0, 0, 0]
category_names = ["Data", "Design", "Plan", "Security", "System", "Web"]

# 카테고리 이름을 인덱스로 매핑
category_index = {category: i for i, category in enumerate(category_names)}

# 색상 지정
colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightblue', 'plum', 'gold']

for i in range(category_length):
    skill = input("Write your skill in English: ")
    user_skills.append(skill)

new_text_tfidf = tfidf_vectorizer.transform(user_skills)  # TF-IDF 벡터로 변환

predicted_categories = loaded_classifier.predict(new_text_tfidf)  # 분류 예측

for category in predicted_categories:
    user_skill_donate[category_index[category]] += 1

percentages = [user_skill_donate[i] / category_length * 100 for i in range(6)]  # percentages 리스트 정의

# 그래프 그리기
plt.figure(figsize=(10, 6))
bars = plt.bar(category_names, percentages, color=colors)  # percentages 리스트를 사용

plt.xlabel('Categories')
plt.ylabel('Percentage (%)')
plt.title('User Skills Distribution by Category')

# 색상을 카테고리에 따라 지정
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.ylim(0, 100)  # y-축 범위 설정
plt.xticks(rotation=45)  # x-축 레이블 회전
plt.show()

plt.figure()

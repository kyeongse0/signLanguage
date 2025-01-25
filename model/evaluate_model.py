import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 저장된 모델 불러오기
model_path = "cnn_lstm_sign_language_model.h5"
model = load_model(model_path)

# 2. 테스트 데이터 로드
X_test = np.load('../data/X_test.npy')
y_test = np.load('../data/y_test.npy')

print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# 3. 모델 평가
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 4. 예측 수행
predictions = model.predict(X_test)

# 5. 혼동 행렬 및 분류 보고서 생성
y_pred = np.argmax(predictions, axis=1)  # 예측값 (클래스 인덱스)
y_true = np.argmax(y_test, axis=1)       # 실제값 (클래스 인덱스)

conf_matrix = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

report = classification_report(y_true, y_pred, target_names=[f"Class {i}" for i in range(y_test.shape[1])])
print("\nClassification Report:")
print(report)

# 6. 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[f"Class {i}" for i in range(y_test.shape[1])], yticklabels=[f"Class {i}" for i in range(y_test.shape[1])])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

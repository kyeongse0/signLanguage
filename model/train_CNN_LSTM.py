import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, TimeDistributed
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 데이터 로드
X_train = np.load('../data/X_train.npy')  # Shape: (samples, timesteps, features)
X_test = np.load('../data/X_test.npy')
y_train = np.load('../data//y_train.npy')
y_test = np.load('../data/y_test.npy')

# CNN+LSTM 모델 정의
model = Sequential()

# TimeDistributed CNN Layers (입력: 시퀀스 내 개별 프레임)
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(30, 64, 64, 1)))  # 예: 프레임 크기 (64x64x1)
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))  # CNN 출력 평탄화

# LSTM Layers (CNN 출력 시퀀스를 시간적으로 학습)
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))

# Fully Connected Layer
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))  # y_train.shape[1]: 클래스 수

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stopping])

# 모델 저장
model.save('cnn_lstm_sign_language_model.h5')

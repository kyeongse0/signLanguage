import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, TimeDistributed
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 데이터 로드
X_train = np.load('../data/X_ztrain.npy')
X_valid = np.load('../data/X_zvalid.npy')
y_train = np.load('../data/y_ztrain.npy')
y_valid = np.load('../data/y_zvalid.npy')

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}")

# CNN+LSTM 모델 정의
model = Sequential()

# TimeDistributed CNN Layers (입력: 시퀀스 내 개별 프레임)
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(30, 64, 64, 1)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))

# LSTM Layers (CNN 출력 시퀀스를 시간적으로 학습)
model.add(LSTM(128))
model.add(Dropout(0.6))  # Dropout 비율 증가

# Fully Connected Layer with L2 Regularization
from tensorflow.keras.regularizers import l2

model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.6))
model.add(Dense(y_train.shape[1], activation='softmax'))

# 모델 컴파일 (RMSprop 옵티마이저 사용)
from tensorflow.keras.optimizers.legacy import RMSprop

optimizer = RMSprop(learning_rate=0.0005)  # 학습률 감소
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # patience 값 조정 및 가중치 복원 추가
checkpoint = ModelCheckpoint('best_model_z.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# 모델 학습
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# 최종 모델 저장 (학습 종료 후 마지막 상태 저장)
model.save('cnn_lstm_sign_language_model_optimized.h5')

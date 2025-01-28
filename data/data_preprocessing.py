import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import random

# 데이터 경로 및 설정
DATA_PATH = 'data'  # 수집된 데이터 경로
GESTURES = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
            'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
            'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ',
            'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ']  # 제스처 목록
SEQUENCES = 90  # 제스처당 시퀀스 개수
FRAMES = 30  # 시퀀스당 프레임 개수
IMAGE_SIZE = (64, 64)  # CNN 입력 크기 (64x64)


# 증강 함수 정의
def add_noise(sequence, noise_level=0.01):
    """시퀀스에 랜덤 노이즈 추가"""
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise



def pad_or_trim_sequence(sequence, target_length):
    """시퀀스를 target_length로 맞추기 위해 패딩하거나 자르기"""
    current_length = len(sequence)
    if current_length < target_length:
        # 부족한 프레임은 제로 패딩으로 채움
        padding = [np.zeros_like(sequence[0]) for _ in range(target_length - current_length)]
        return sequence + padding
    elif current_length > target_length:
        # 초과하는 프레임은 잘라냄
        return sequence[:target_length]
    return sequence


# 데이터 로드 및 전처리
X, y_labels = [], []  # y 대신 y_labels로 이름 변경

for idx, gesture in enumerate(GESTURES):
    gesture_path = os.path.join(DATA_PATH, gesture)
    for sequence in range(SEQUENCES):
        sequence_path = os.path.join(gesture_path, f"{sequence}.npy")
        frames = np.load(sequence_path)  # Shape: (FRAMES, 63)

        # 각 프레임을 CNN 입력 크기에 맞게 변환 (2D 이미지로 reshape)
        frames_reshaped = []
        for frame in frames:
            # 랜드마크 데이터를 (21, 3) -> (64, 64)로 매핑
            frame_image = np.zeros(IMAGE_SIZE)  # 빈 이미지 생성
            for i in range(21):  # 손 랜드마크의 각 점
                x_coord, y_coord, z_coord = frame[i * 3:(i + 1) * 3]  # 변수명 변경
                px, py = int(x_coord * IMAGE_SIZE[0]), int(y_coord * IMAGE_SIZE[1])
                if 0 <= px < IMAGE_SIZE[0] and 0 <= py < IMAGE_SIZE[1]:
                    frame_image[py, px] = z_coord + 1.0  # 깊이 값을 추가해 표현

            frames_reshaped.append(frame_image)

        # 시퀀스 길이를 FRAMES로 맞춤
        frames_reshaped = pad_or_trim_sequence(frames_reshaped, FRAMES)

        X.append(frames_reshaped)
        y_labels.append(idx)  # y 대신 y_labels 사용

        # 증강된 데이터 추가
        augmented_frames = add_noise(frames)
        augmented_frames_reshaped = []
        for frame in augmented_frames:
            frame_image_augmented = np.zeros(IMAGE_SIZE)
            for i in range(21):
                x_coord, y_coord, z_coord = frame[i * 3:(i + 1) * 3]  # 변수명 변경
                px, py = int(x_coord * IMAGE_SIZE[0]), int(y_coord * IMAGE_SIZE[1])
                if 0 <= px < IMAGE_SIZE[0] and 0 <= py < IMAGE_SIZE[1]:
                    frame_image_augmented[py, px] = z_coord + 1.0

            augmented_frames_reshaped.append(frame_image_augmented)

        # 증강된 시퀀스도 길이를 FRAMES로 맞춤
        augmented_frames_reshaped = pad_or_trim_sequence(augmented_frames_reshaped, FRAMES)

        X.append(augmented_frames_reshaped)
        y_labels.append(idx)  # y 대신 y_labels 사용

# 배열로 변환 및 차원 추가 (CNN 채널 추가: 흑백 이미지는 채널=1)
X = np.array(X).astype('float32')  # Shape: (samples, FRAMES, IMAGE_SIZE[0], IMAGE_SIZE[1])
X = X[..., np.newaxis]             # Shape: (samples, FRAMES, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
y_labels = to_categorical(y_labels).astype(int)  # One-hot 인코딩

# 데이터 분리 (7:2:1 - train:validation:test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_labels, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# 데이터 저장
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('X_valid.npy', X_valid)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
np.save('y_valid.npy', y_valid)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
import cv2
import numpy as np
import mediapipe as mp
import random
import time
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image  # Pillow 라이브러리

# 1. 모델 및 설정 로드
model_path = "cnn_lstm_sign_language_model_optimized_4.h5"  # 저장된 모델 경로
model = load_model(model_path)

GESTURES = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
            'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
            'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ',
            'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ']  # 제스처 목록

SEQUENCE_LENGTH = 30  # 시퀀스 길이 (모델 입력)
IMAGE_SIZE = (64, 64)  # 입력 이미지 크기 (모델 입력)
TIME_LIMIT = 10  # 퀴즈 시간 제한 (초)

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 한글 폰트 설정 (NanumGothic.ttf 또는 다른 한글 폰트를 사용)
font_path = "/System/Library/Fonts/Supplemental/AppleSDGothicNeo.ttc"  # macOS 기준
font = ImageFont.truetype(font_path, 32)  # 폰트 크기 설정


# 2. 퀴즈 설정
def get_new_quiz():
    """랜덤으로 새로운 퀴즈 문제를 선택"""
    return random.choice(GESTURES), time.time()  # 문제와 시작 시간 반환


current_quiz, start_time = get_new_quiz()  # 첫 문제 선정
sequence = []  # 시퀀스를 저장할 리스트
feedback_text = ""  # 사용자 피드백 메시지 (맞았는지 여부)
feedback_time = 0  # 피드백 메시지가 표시된 시간

# 웹캠 실행
cap = cv2.VideoCapture(0)


def preprocess_landmarks(landmarks):
    """손 랜드마크를 (64x64) 이미지로 변환"""
    frame_image = np.zeros(IMAGE_SIZE, dtype=np.float32)  # 빈 이미지 생성
    for lm in landmarks:
        px, py = int(lm.x * IMAGE_SIZE[0]), int(lm.y * IMAGE_SIZE[1])
        if 0 <= px < IMAGE_SIZE[0] and 0 <= py < IMAGE_SIZE[1]:
            frame_image[py, px] = 1  # z 값 없이 좌표만 사용
    return frame_image


with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 현재 시간 확인
        elapsed_time = time.time() - start_time
        remaining_time = max(0, TIME_LIMIT - int(elapsed_time))  # 음수가 되지 않도록 제한

        predicted_gesture = None  # 매 프레임마다 초기화

        # BGR -> RGB 변환 및 MediaPipe 처리
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # 랜드마크 추출 및 시각화
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 랜드마크를 64x64 이미지로 변환
                frame_image = preprocess_landmarks(hand_landmarks.landmark)
                sequence.append(frame_image)

                # 시퀀스 길이가 SEQUENCE_LENGTH를 초과하면 가장 오래된 프레임 제거
                if len(sequence) > SEQUENCE_LENGTH:
                    sequence.pop(0)

                # 시퀀스가 충분히 쌓이면 예측 수행
                if len(sequence) == SEQUENCE_LENGTH:
                    model_input = np.array(sequence)  # 리스트 -> numpy 배열 변환
                    model_input = np.expand_dims(model_input, axis=0)  # 배치 차원 추가
                    model_input = model_input[..., np.newaxis]  # 채널 추가

                    predictions = model.predict(model_input)
                    predicted_gesture_index = np.argmax(predictions)
                    predicted_gesture = GESTURES[predicted_gesture_index]

                    # 정답 확인
                    if predicted_gesture == current_quiz:
                        feedback_text = " 맞았습니다~ 다음 문제로 이동합니다."
                        feedback_time = time.time()  # 피드백 표시 시작 시간
                        current_quiz, start_time = get_new_quiz()  # 새로운 문제 출제
                        sequence = []  # 시퀀스 초기화
                    else:
                        feedback_text = " 더 노력해보세요!"
                        feedback_time = time.time()

        # 시간 초과 시 새로운 문제로 이동
        if elapsed_time > TIME_LIMIT:
            feedback_text = " 시간 초과! 다음 문제로 이동합니다."
            feedback_time = time.time()
            current_quiz, start_time = get_new_quiz()
            sequence = []  # 시퀀스 초기화

        # 결과 출력: OpenCV 이미지를 Pillow 이미지로 변환하여 한글 출력
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # ⏳ 남은 시간 별도로 출력
        draw.text((10, 10), f" 남은 시간: {remaining_time}초", font=font, fill=(0, 255, 255))

        # 📝 현재 문제 표시
        draw.text((10, 50), f" 문제: {current_quiz}", font=font, fill=(255, 255, 255))

        # 🎯 피드백 문구 출력 (1초 유지)
        if feedback_text and (time.time() - feedback_time <= 1):
            draw.text((10, 100), feedback_text, font=font, fill=(255, 0, 0) if "!" in feedback_text else (0, 255, 0))

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # 화면 출력
        cv2.imshow('Sign Language Quiz', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):  # q 키를 누르면 종료
            break

cap.release()
cv2.destroyAllWindows()

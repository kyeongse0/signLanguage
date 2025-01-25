import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image  # Pillow 라이브러리

# 1. 모델 및 설정 로드
model_path = "cnn_lstm_sign_language_model.h5"  # 저장된 모델 경로
model = load_model(model_path)

GESTURES = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
            'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
            'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ',
            'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ']  # 제스처 목록

SEQUENCE_LENGTH = 30  # 시퀀스 길이 (모델 입력)
IMAGE_SIZE = (64, 64)  # 입력 이미지 크기 (모델 입력)

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 한글 폰트 설정 (NanumGothic.ttf 또는 다른 한글 폰트를 사용)
font_path = "/System/Library/Fonts/Supplemental/AppleSDGothicNeo.ttc"  # 폰트 파일 경로를 지정하세요
font = ImageFont.truetype(font_path, 32)  # 폰트 크기 설정

# 2. 실시간 웹캠 처리
cap = cv2.VideoCapture(0)  # 웹캠 열기
sequence = []  # 시퀀스를 저장할 리스트
predicted_gesture = None

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB 변환 및 MediaPipe 처리
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # 랜드마크 추출 및 시각화
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 랜드마크를 (21, 3) 형태로 변환 후 (64x64) 이미지로 매핑
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                frame_image = np.zeros(IMAGE_SIZE)  # 빈 이미지 생성
                for i, (x, y, z) in enumerate(landmarks):
                    px, py = int(x * IMAGE_SIZE[0]), int(y * IMAGE_SIZE[1])
                    if 0 <= px < IMAGE_SIZE[0] and 0 <= py < IMAGE_SIZE[1]:
                        frame_image[py, px] = z + 1.0

                # 시퀀스에 추가
                sequence.append(frame_image)

                # 시퀀스 길이가 SEQUENCE_LENGTH를 초과하면 가장 오래된 프레임 제거
                if len(sequence) > SEQUENCE_LENGTH:
                    sequence.pop(0)

                # 시퀀스가 충분히 쌓이면 예측 수행
                if len(sequence) == SEQUENCE_LENGTH:
                    model_input = np.expand_dims(sequence,
                                                 axis=0)  # 모델 입력 형식으로 변환 (1, SEQUENCE_LENGTH, IMAGE_SIZE[0], IMAGE_SIZE[1])
                    model_input = model_input[
                        ..., np.newaxis]  # 채널 추가 (1, SEQUENCE_LENGTH, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
                    predictions = model.predict(model_input)
                    predicted_gesture_index = np.argmax(predictions)
                    predicted_gesture = GESTURES[predicted_gesture_index]

        # 결과 출력: OpenCV 이미지를 Pillow 이미지로 변환하여 한글 출력
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        text = f"예측된 제스처: {predicted_gesture}" if predicted_gesture else "제스처를 기다리는 중..."
        draw.text((10, 10), text, font=font, fill=(255, 255, 255))  # 흰색 텍스트 출력

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # 화면 출력
        cv2.imshow('Real-Time Sign Language Recognition', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):  # q 키를 누르면 종료
            break

cap.release()
cv2.destroyAllWindows()

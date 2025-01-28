import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 데이터 저장 경로 및 제스처 설정
DATA_PATH = 'data'  # 데이터 저장 경로
GESTURES = ['ㅉ']  # 제스처 목록
'''
GESTURES = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
            'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
            'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ',
            'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ']  # 제스처 목록
'''
SEQUENCES = 30  # 제스처당 시퀀스 개수
FRAMES = 30  # 시퀀스당 프레임 개수

# 한글 폰트 설정
FONT_PATH = "/System/Library/Fonts/Supplemental/AppleSDGothicNeo.ttc"  # 한글 폰트 파일 경로
font = ImageFont.truetype(FONT_PATH, 32)  # 폰트 크기 설정

# 데이터 저장 디렉토리 생성
os.makedirs(DATA_PATH, exist_ok=True)
for gesture in GESTURES:
    os.makedirs(os.path.join(DATA_PATH, gesture), exist_ok=True)

# 카메라 캡처 시작
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    for gesture in GESTURES:
        # 이미 존재하는 파일 개수 확인
        gesture_path = os.path.join(DATA_PATH, gesture)
        existing_files = len([f for f in os.listdir(gesture_path) if f.endswith('.npy')])

        for sequence in range(existing_files, existing_files + SEQUENCES):  # 이어서 저장
            frames = []
            print(f"Collecting data for {gesture}, sequence {sequence + 1}/{existing_files + SEQUENCES}")

            for frame_num in range(FRAMES):
                ret, frame = cap.read()
                if not ret:
                    break

                # BGR -> RGB 변환 및 처리
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True

                # 랜드마크 추출
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                        frames.append(landmarks)
                else:
                    frames.append(np.zeros(21 * 3))  # 손이 감지되지 않을 경우 빈 값 추가

                # OpenCV 이미지를 Pillow 이미지로 변환
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)

                # 현재 제스처와 진행 상태 표시 (한글 출력)
                text = f"제스처: {gesture} | 시퀀스: {sequence + 1}/{existing_files + SEQUENCES} | 프레임: {frame_num + 1}/{FRAMES}"
                draw.text((10, 10), text, font=font, fill=(255, 255, 255))  # 흰색 텍스트

                # Pillow 이미지를 다시 OpenCV 이미지로 변환
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                # 화면 출력
                cv2.imshow('Data Collection - Press Q to Quit', frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # 시퀀스 데이터 저장
            np.save(os.path.join(DATA_PATH, gesture, f"{sequence}.npy"), np.array(frames))

cap.release()
cv2.destroyAllWindows()

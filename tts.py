from gtts import gTTS
from playsound import playsound

def speak_caption():
    # 캡션 텍스트 가져오기
    text = cap.get()
    
    if text.strip() == "":
        print("캡션이 비어 있습니다.")
        return

    try:
        # 텍스트를 음성으로 변환하여 저장
        tts = gTTS(text=text, lang='ko')
        tts.save("caption.mp3")
        
        # 음성 재생
        playsound("caption.mp3")
        
    except Exception as e:
        print(f"음성 출력 실패: {e}")

# GUI 버튼과 연결하는 부분
capTTS.config(command=speak_caption)
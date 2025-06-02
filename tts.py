import time
from gtts import gTTS
from playsound import playsound

def speak_caption(text):
    if text.strip() == "":
        print("캡션이 비어 있습니다.")
        return

    try:
        filename = f"caption_{int(time.time()*1000)}.mp3"  # 타임스탬프 붙인 파일명
        tts = gTTS(text=text, lang='ko')
        tts.save(filename)
        playsound(filename)
        # 재생 끝난 후 삭제
        import os
        os.remove(filename)

    except Exception as e:
        print(f"음성 출력 실패: {e}")


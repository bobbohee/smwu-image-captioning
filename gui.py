import tkinter as tk
import urllib.request
from tts import speak_caption
from PIL import Image, ImageTk

frame = tk.Tk()
frame.title("눈뜬송이")
frame.geometry("1200x700")
label = tk.Label(frame)


#URL 입력받기
urlLabel = tk.Label(frame, text="이미지 URL 입력", font=("맑은 고딕", 12))
urldown = tk.Button(frame, text="시작", command=lambda: download_image())
urlinput = tk.Entry(frame, width=100)


#URL 이미지 로컬에 다운로드
def download_image():
    url = urlinput.get()
    save = "image.png"

    try:
        with urllib.request.urlopen(url) as response:
            with open(save, 'wb') as out_file:
               out_file.write(response.read())

        # 이미지 출력
        img = Image.open(save)
        original_width, original_height = img.size
        target_height = 450
        target_width = int(original_width * (target_height / original_height))
        img = img.resize((target_width, target_height), Image.LANCZOS)

        img_v2 = ImageTk.PhotoImage(img)
        label.config(image=img_v2)
        label.image = img_v2

        cap.config(state='normal')
        cap.delete(0, tk.END)
        cap.insert(0, "우왕 캡션")  #여기에 캡션 넣기
        cap.config(state='disabled')

        #TTS 출력 추가

    except Exception as e:
        print(f"이미지 다운로드 실패: {e}")
        label.config(text="이미지 다운로드 실패", image='')



#캡션&TTS 출력
capLabel = tk.Label(frame, text="캡션", font=("맑은 고딕", 12))
btnImg = tk.PhotoImage(file="TTSbutton.png")

cap = tk.Entry(frame, width=100)
cap.config(state='disabled')

capTTS = tk.Button(frame, image=btnImg, command=lambda: speak_caption(cap.get()))




urlLabel.pack()
urldown.pack()
urlinput.pack()
label.pack()
capLabel.pack(pady=(30,0))
capTTS.pack()
cap.pack()
frame.mainloop()
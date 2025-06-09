import asyncio
from googletrans import Translator

# 비동기 번역 함수
async def translate_kor_to_eng(text):
    translator = Translator()
    result = await translator.translate(text, src='en', dest='ko')
    return result.text

# 동기 함수: asyncio.run으로 비동기 함수 실행
def translate_text(text):
    return asyncio.run(translate_kor_to_eng(text))

#번역모듈
def translate_caption(text):
    try:
        translated_text = translate_text(text)
        return translated_text
    except Exception as e:
        print(f"번역 실패: {e}")
        return "번역 실패"
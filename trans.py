import saved_model
from deep_translator import GoogleTranslator

def translate_caption():
    try:
        # 모델 캡션 생성
        text = saved_model.generate_and_display_caption(
            image_path='image.png', 
            model_path='model.keras',
            tokenizer_path='tokenizer.pkl',
            feature_extractor_path='feature_extractor.keras',
            max_length=34,
            img_size=224)

        # 영어 -> 한국어 번역
        translator = GoogleTranslator(source='en', target='ko')
        result = translator.translate(text)
        return result
        
    except Exception as e:
        print(f"번역 실패: {e}")
        return "번역 실패"
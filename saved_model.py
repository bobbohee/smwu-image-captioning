import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

path = os.getcwd()

# load save files
model_path = 'model.keras'
tokenizer_path = 'tokenizer.pkl'
feature_extractor_path = 'feature_extractor.keras'


def generate_and_display_caption(image_path, model_path, tokenizer_path, feature_extractor_path, max_length=21, img_size=224):
    # Load the trained models and tokenizer
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)

    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
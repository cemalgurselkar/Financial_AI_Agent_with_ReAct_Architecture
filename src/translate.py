from deep_translator import GoogleTranslator

class TranslatorWrapper:
    def __init__(self):
        self.to_en = GoogleTranslator(source="tr", target="en")
        self.to_tr = GoogleTranslator(source="en", target="tr")
    
    def translate_to_tr(self, text):
        try:
            return self.to_tr.translate(text)
        except Exception as e:
            print(f"Error: {e}")
            return text
    
    def translate_to_en(self, text):
        try:
            return self.to_en.translate(text)
        except Exception as e:
            print(f"Error: {e}")
            return text
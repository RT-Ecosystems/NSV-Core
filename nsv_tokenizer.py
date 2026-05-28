import re

class NSV_Tokenizer:
    def __init__(self):
        pass

    def tokenize(self, text):
        """टेक्स्ट को साफ करके लोअरकेस शब्दों और कैरेक्टर एन-ग्राम्स में तोड़ता है।"""
        text = text.lower().strip()
        # सिर्फ अक्षरों और नंबरों को रखना
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        return words

    def get_char_trigrams(self, word):
        """एक शब्द को 3-3 अक्षरों के टुकड़ों (Trigrams) में तोड़ता है ताकि बनावट समझ आए।"""
        # उदाहरण: "cat" -> ["#ca", "cat", "at#"]
        padded = f"#{word}#"
        return [padded[i:i+3] for i in range(len(padded)-2)]
      

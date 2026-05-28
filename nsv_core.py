import numpy as np
from .nsv_tokenizer import NSV_Tokenizer

class NSV_Core:
    def __init__(self, base_dim=10000):
        self.tokenizer = NSV_Tokenizer()
        self.base_dim = base_dim
        self.alphabet_vectors = {}
        print(f"🧬 NSV सेमेंटिक कोर चालू! बेस डायमेंशन: {base_dim}")

    def _get_char_vector(self, char):
        """हर अक्षर या ट्रिग्राम के लिए एक फिक्स (Deterministic) हाइपरवेक्टर बनाना।"""
        if char not in self.alphabet_vectors:
            # अक्षर के नाम को सीड बनाकर हमेशा एक जैसा वेक्टर पाना (No randomness)
            seed = abs(hash(char)) % (2**32)
            np.random.seed(seed)
            self.alphabet_vectors[char] = np.random.choice([-1, 1], size=self.base_dim)
        return self.alphabet_vectors[char]

    def generate_neuro_print(self, input_text):
        """
        अक्षरों के वेक्टर्स को शिफ्ट (Permute) और गुणा (XOR/Multiply) करके
        पूरे टेक्स्ट का एक 'अर्थ समझने वाला' न्यूरो-प्रिंट बनाता है।
        """
        words = self.tokenizer.tokenize(input_text)
        if not words:
            return np.ones(self.base_dim), self.base_dim

        # पूरे वाक्य का मास्टर वेक्टर
        text_vector = np.zeros(self.base_dim)

        for word in words:
            word_vector = np.ones(self.base_dim)
            trigrams = self.tokenizer.get_char_trigrams(word)
            
            for i, trigram in enumerate(trigrams):
                vec = self._get_char_vector(trigram)
                # रोल (Roll) करना यानी अक्षर की पोजीशन के हिसाब से वेक्टर को शिफ्ट करना
                # इससे "tea" और "eat" का फर्क मॉडल समझ जाता है
                shifted_vec = np.roll(vec, shift=i)
                # बाइंडिंग (गुणा करना)
                word_vector = word_vector * shifted_vec
            
            # बंडलिंग (जोड़ना) - सारे शब्दों के वेक्टर्स को वाक्य में मिलाना
            text_vector += word_vector

        # अंत में वेक्टर को वापस -1 और 1 के फॉर्म में लाना
        semantic_print = np.sign(text_vector)
        semantic_print[semantic_print == 0] = 1
        
        return semantic_print, self.base_dim
        

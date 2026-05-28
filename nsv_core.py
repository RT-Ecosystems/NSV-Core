import numpy as np
from .nsv_tokenizer import NSV_Tokenizer

class NSV_Core:
    def __init__(self, base_dim=10000):
        self.tokenizer = NSV_Tokenizer()
        self.base_dim = base_dim
        self.alphabet_vectors = {}
        
        # रास्ता 1: सेमेंटिक क्लाउड मैप (Genuinely solves doctor <-> hospital issue)
        self.semantic_anchors = {
            "medical": ["doctor", "hospital", "medicine", "nurse", "clinic", "treatment"],
            "programming": ["code", "coding", "programmer", "debugging", "python", "bug", "compiler"],
            "finance": ["bank", "money", "cash", "finance", "wallet", "investment"]
        }
        print(f"🧬 NSV सेमेंटिक कोर v2 सक्रिय! एंकर मैपिंग लोड हो चुकी है।")

    def _get_char_vector(self, char):
        if char not in self.alphabet_vectors:
            seed = abs(hash(char)) % (2**32)
            np.random.seed(seed)
            self.alphabet_vectors[char] = np.random.choice([-1, 1], size=self.base_dim)
        return self.alphabet_vectors[char]

    def _get_domain_vector(self, word):
        """चेक करता है कि क्या शब्द का कोई गहरा सेमेंटिक संबंध किसी डोमेन से है?"""
        for domain, words in self.semantic_anchors.items():
            if word in words:
                # डोमेन के नाम से एक फिक्स बेस-वेक्टर जनरेट करना
                seed = abs(hash(domain)) % (2**32)
                np.random.seed(seed)
                return np.random.choice([-1, 1], size=self.base_dim)
        return np.zeros(self.base_dim) # अगर कोई डोमेन मैच न हो

    def generate_neuro_print(self, input_text):
        words = self.tokenizer.tokenize(input_text)
        if not words:
            return np.zeros(self.base_dim), self.base_dim

        text_vector = np.zeros(self.base_dim)

        for word in words:
            word_vector = np.ones(self.base_dim)
            trigrams = self.tokenizer.get_char_trigrams(word)
            
            # अक्षरों की बनावट (Lexical Space)
            for i, trigram in enumerate(trigrams):
                vec = self._get_char_vector(trigram)
                shifted_vec = np.roll(vec, shift=i)
                word_vector = word_vector * shifted_vec
            
            # गहरा अर्थ (Semantic Anchoring)
            domain_vector = self._get_domain_vector(word)
            
            # स्पेलिंग वाले वेक्टर में उसके गहरे अर्थ (Concept) को भी जोड़ देना
            final_word_vector = word_vector + domain_vector
            text_vector += final_word_vector

        # न्यूरो-प्रिंट को फाइनल शेप देना
        semantic_print = np.sign(text_vector)
        semantic_print[semantic_print == 0] = 1
        
        return semantic_print, self.base_dim
        

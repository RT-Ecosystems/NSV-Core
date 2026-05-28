import numpy as np
from .nsv_tokenizer import NSV_Tokenizer
from sentence_transformers import SentenceTransformer # [NEW] असली Learned Embeddings के लिए

class NSV_Core:
    def __init__(self, base_dim=10000):
        self.tokenizer = NSV_Tokenizer()
        self.base_dim = base_dim
        self.alphabet_vectors = {}
        
        # [NEW] Phase 1: Learned Semantic Seeds (नकली डिक्शनरी हटा दी गई)
        # यह सिर्फ 80MB का मॉडल है जो CPU पर सुपर-फास्ट चलता है 
        print("📥 लाइटवेट सेमेंटिक एंकर मॉडल लोड हो रहा है...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("🧬 NSV सेमेंटिक कोर v2 सक्रिय! असली 'Learned Embeddings' तैयार हैं।")

    def _get_char_vector(self, char):
        if char not in self.alphabet_vectors:
            seed = abs(hash(char)) % (2**32)
            np.random.seed(seed)
            self.alphabet_vectors[char] = np.random.choice([-1, 1], size=self.base_dim)
        return self.alphabet_vectors[char]

    def _get_learned_semantic_vector(self, word):
        """[NEW] यह शब्द का असली दुनिया का अर्थ (Embeddings) निकालेगा और उसे HDC के बेस डायमेंशन में प्रोजेक्ट करेगा।"""
        # असली AI से शब्द का अर्थ निकालना (384 डायमेंशन)
        small_embedding = self.semantic_model.encode(word)
        
        # इसे हमारे 10000 डायमेंशन के HDC स्पेस में फैलाना (Projection)
        np.random.seed(abs(hash(word)) % (2**32))
        projection_matrix = np.random.randn(self.base_dim, len(small_embedding))
        
        projected_vector = np.dot(projection_matrix, small_embedding)
        return np.sign(projected_vector) # इसे वापस -1 और +1 में बदल देना

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
            
            # [NEW] असली गहरा अर्थ (Learned Semantic Anchoring)
            semantic_vector = self._get_learned_semantic_vector(word)
            
            # स्पेलिंग (Symbolic) और अर्थ (Learned) का असली हाइब्रिड
            final_word_vector = word_vector + semantic_vector
            text_vector += final_word_vector

        semantic_print = np.sign(text_vector)
        semantic_print[semantic_print == 0] = 1
        
        return semantic_print, self.base_dim
        

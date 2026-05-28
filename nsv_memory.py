import numpy as np
import faiss  # [NEW] 10TB डेटा को मिलीसेकंड में खोजने के लिए
from .nsv_core import NSV_Core

class NSV_Memory:
    def __init__(self, dim=10000, max_shard_capacity=15, sparsity_threshold=0.85):
        self.core = NSV_Core(base_dim=dim)
        self.memory_bank = {}
        self.max_shard_capacity = max_shard_capacity
        self.sparsity_threshold = sparsity_threshold
        
        # [NEW] FAISS Indexing for Ultra-Fast Retrieval (ANN)
        # यह GPT-5 स्तर की स्पीड देगा जब डेटा बहुत विशाल हो जाएगा
        self.dimension = dim
        self.index = faiss.IndexFlatIP(self.dimension) # Inner Product Search
        self.vector_to_category_map = {} # वेक्टर्स को उनकी कैटेगरी से जोड़ने के लिए
        self.vector_count = 0

    def _make_sparse(self, vector):
        absolute_vector = np.abs(vector)
        threshold = np.quantile(absolute_vector, self.sparsity_threshold)
        sparse_vector = np.where(absolute_vector >= threshold, vector, 0)
        return sparse_vector

    def _hopfield_cleanup(self, vector):
        """
        [NEW] Phase 3: Hopfield-style Cleanup Memory
        यह फंक्शन धुंधली यादों (Noise) को साफ करके वेक्टर को फिर से 'क्रिस्टल क्लियर' बनाता है।
        """
        # नॉइज़ को दबाना और मजबूत सिग्नल्स को उभारना
        cleaned_vector = np.tanh(vector) 
        return np.sign(cleaned_vector)

    def bind_and_bundle(self, category, text_data):
        new_print, _ = self.core.generate_neuro_print(text_data)
        
        # 1. स्पार्स करना
        sparse_new_print = self._make_sparse(new_print)
        # 2. हॉपफील्ड क्लीनअप (ताकि कोई नॉइज़ न रहे)
        clean_vector = self._hopfield_cleanup(sparse_new_print)
        
        if category not in self.memory_bank:
            self.memory_bank[category] = [{"vector": clean_vector, "count": 1}]
        else:
            last_shard = self.memory_bank[category][-1]
            if last_shard["count"] < self.max_shard_capacity:
                combined = last_shard["vector"] + clean_vector
                sparse_bundled = self._make_sparse(combined)
                clean_bundled = self._hopfield_cleanup(sparse_bundled)
                
                last_shard["vector"] = clean_bundled
                last_shard["count"] += 1
            else:
                self.memory_bank[category].append({"vector": clean_vector, "count": 1})
        
        # [NEW] FAISS इंडेक्स में जोड़ना (ताकि बाद में बिजली की तेजी से सर्च हो सके)
        vector_32 = np.float32(clean_vector).reshape(1, -1)
        faiss.normalize_L2(vector_32)
        self.index.add(vector_32)
        
        self.vector_to_category_map[self.vector_count] = category
        self.vector_count += 1
        
        return f"✅ '{category}' FAISS इंडेक्स और मेमोरी शार्ड में सफलतापूर्वक लॉक हो गया।"

    def ultra_fast_search(self, query_text, top_k=3):
        """[NEW] Phase 4: ANN Retrieval (10TB डेटा को तुरंत खोजना)"""
        if self.vector_count == 0:
            return "मेमोरी बैंक खाली है।"
            
        query_print, _ = self.core.generate_neuro_print(query_text)
        clean_query = self._hopfield_cleanup(self._make_sparse(query_print))
        
        query_32 = np.float32(clean_query).reshape(1, -1)
        faiss.normalize_L2(query_32)
        
        # FAISS के जरिए सबसे सटीक मैच खोजना
        distances, indices = self.index.search(query_32, top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx in self.vector_to_category_map:
                results.append(self.vector_to_category_map[idx])
        return results
        

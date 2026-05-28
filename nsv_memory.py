import numpy as np
from .nsv_core import NSV_Core

class NSV_Memory:
    def __init__(self, dim=10000, max_shard_capacity=15, sparsity_threshold=0.85):
        self.core = NSV_Core(base_dim=dim)
        self.memory_bank = {}
        self.max_shard_capacity = max_shard_capacity
        self.sparsity_threshold = sparsity_threshold # 85% हिस्से को 0 (शांत) रखने के लिए

    def _make_sparse(self, vector):
        """रास्ता 2: सघन वेक्टर को स्पार्स हाइपरवेक्टर (Sparse Vector) में बदलना"""
        # यह केवल टॉप एक्टिव सिग्नल्स को रखेगा और बाकी सबको 0 कर देगा
        absolute_vector = np.abs(vector)
        threshold = np.quantile(absolute_vector, self.sparsity_threshold)
        
        sparse_vector = np.where(absolute_vector >= threshold, vector, 0)
        return sparse_vector

    def bind_and_bundle(self, category, text_data):
        new_print, _ = self.core.generate_neuro_print(text_data)
        # सेव करने से पहले ही वेक्टर को स्पार्स (हल्का) कर दो
        sparse_new_print = self._make_sparse(new_print)
        
        if category not in self.memory_bank:
            self.memory_bank[category] = [{"vector": sparse_new_print, "count": 1}]
            return f"✅ '{category}' में पहला स्पार्स शार्ड (Sparse Shard-0) सुरक्षित सेव हुआ।"
        
        last_shard = self.memory_bank[category][-1]
        
        if last_shard["count"] < self.max_shard_capacity:
            # बंडलिंग के दौरान स्पार्सिटी को बनाए रखना (No memory collapse)
            combined = last_shard["vector"] + sparse_new_print
            sparse_bundled = self._make_sparse(combined)
            
            last_shard["vector"] = sparse_bundled
            last_shard["count"] += 1
            return f"🔄 Shard-{len(self.memory_bank[category])-1} में एंटी-नॉइज़ बंडलिंग सफल।"
        else:
            # नया शार्ड जनरेशन
            self.memory_bank[category].append({"vector": sparse_new_print, "count": 1})
            return f"🚀 एंटी-कैटास्ट्रॉफी ट्रिगर! नया स्पार्स शार्ड (Shard-{len(self.memory_bank[category])-1}) बनाया गया।"
            

import numpy as np
from .nsv_core import NSV_Core

class NSV_Memory:
    def __init__(self, dim=10000, max_shard_capacity=15):
        self.core = NSV_Core(base_dim=dim)
        # अब यह फ्लैट नहीं, बल्कि हियरार्किकल शार्ड्स स्टोर करेगा: {category: [list of shards]}
        self.memory_bank = {}
        # सुपरपोजीशन कैटास्ट्रॉफी से बचने के लिए एक शार्ड में अधिकतम 15 वेक्टर्स ही बंडल होंगे
        self.max_shard_capacity = max_shard_capacity 

    def bind_and_bundle(self, category, text_data):
        """नई जानकारी के न्यूरो-प्रिंट को हियरार्किकल शार्ड्स में सुरक्षित बंडल करता है।"""
        new_print, _ = self.core.generate_neuro_print(text_data)
        
        # अगर कैटेगरी पहली बार आई है
        if category not in self.memory_bank:
            self.memory_bank[category] = [{"vector": new_print, "count": 1}]
            return f"✅ '{category}' में पहला हियरार्किकल शार्ड (Shard-0) तैयार।"
        
        # चैटजीपीटी के 'Memory Saturation' का तोड़:
        # हम चेक करेंगे कि क्या आखिरी शार्ड अपनी क्षमता (Capacity) पार कर चुका है?
        last_shard = self.memory_bank[category][-1]
        
        if last_shard["count"] < self.max_shard_capacity:
            # क्षमता के अंदर है, तो सुरक्षित बंडलिंग करो
            combined = last_shard["vector"] + new_print
            bundled_print = np.sign(combined)
            bundled_print[bundled_print == 0] = 1 
            
            last_shard["vector"] = bundled_print
            last_shard["count"] += 1
            return f"🔄 '{category}' के Shard-{len(self.memory_bank[category])-1} में डेटा बंडल हुआ (Count: {last_shard['count']})"
        else:
            # क्षमता पूरी हो गई! नया शार्ड (New Shard) बनाओ ताकि पुराना डेटा क्रैश न हो
            self.memory_bank[category].append({"vector": new_print, "count": 1})
            return f"🚀 एंटी-कैटास्ट्रॉफी ट्रिगर! '{category}' में नया शार्ड (Shard-{len(self.memory_bank[category])-1}) बनाया गया।"
            

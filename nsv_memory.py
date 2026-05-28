import numpy as np
from .nsv_core import NSV_Core

class NSV_Memory:
    def __init__(self, dim=10000):
        self.core = NSV_Core(base_dim=dim)
        self.memory_bank = {}

    def bind_and_bundle(self, category, text_data):
        """नई जानकारी के न्यूरो-प्रिंट को पुरानी मेमोरी में ठूंसेगा।"""
        new_print, _ = self.core.generate_neuro_print(text_data)
        
        if category not in self.memory_bank:
            self.memory_bank[category] = new_print
        else:
            # पुरानी और नई मेमोरी को आपस में जोड़ना (Associative Bundling)
            combined = self.memory_bank[category] + new_print
            bundled_print = np.sign(combined)
            bundled_print[bundled_print == 0] = 1 
            self.memory_bank[category] = bundled_print
            
        return f"✅ '{category}' में सेमेंटिक डेटा बंडल हो गया।"
        

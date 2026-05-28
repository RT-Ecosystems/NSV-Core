import numpy as np
from .nsv_core import NSV_Core

class NSV_Memory:
    def __init__(self):
        self.core = NSV_Core()
        self.memory_bank = {}

    def bind_and_bundle(self, category, text_data):
        """नई जानकारी को मौजूदा मेमोरी में बंडल (Bundle) करता है।"""
        new_print, dim = self.core.generate_neuro_print(text_data)
        
        if category not in self.memory_bank:
            self.memory_bank[category] = new_print
        else:
            existing_print = self.memory_bank[category]
            
            # अगर डायमेंशन अलग हैं, तो छोटे वाले को बड़े के बराबर करना (Padding)
            max_dim = max(len(existing_print), len(new_print))
            if len(existing_print) < max_dim:
                existing_print = np.pad(existing_print, (0, max_dim - len(existing_print)), 'constant')
            if len(new_print) < max_dim:
                new_print = np.pad(new_print, (0, max_dim - len(new_print)), 'constant')
                
            # बंडलिंग (जोड़ना और साइन निकालना)
            combined = existing_print + new_print
            bundled_print = np.sign(combined)
            # अगर 0 हो जाए, तो उसे 1 कर दें
            bundled_print[bundled_print == 0] = 1 
            self.memory_bank[category] = bundled_print
            
        return f"✅ '{category}' में डेटा बंडल हो गया। (साइज: {len(self.memory_bank[category])})"
      

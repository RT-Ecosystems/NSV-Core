class NSV_Streamer:
    def __init__(self, memory_module):
        self.memory = memory_module

    def train_from_stream(self, category, data_generator, max_items=None):
        """
        यूजर द्वारा दिए गए डेटा स्ट्रीम (Generator) से मॉडल को ट्रेन करता है।
        यूजर तय करेगा कि डेटा कहाँ से आएगा और कब तक चलेगा।
        """
        print(f"🚀 '{category}' के लिए डेटा स्ट्रीमिंग शुरू...")
        count = 0
        
        for data_chunk in data_generator:
            if max_items and count >= max_items:
                break
                
            # डेटा को सीधे मेमोरी में ठूंसना
            self.memory.bind_and_bundle(category, data_chunk)
            count += 1
            
            if count % 1000 == 0:
                print(f"🔄 {count} डेटा चंक्स प्रोसेस हो गए...")
                
        print(f"✅ स्ट्रीमिंग पूरी हुई! कुल {count} चंक्स प्रोसेस किए गए।")
      

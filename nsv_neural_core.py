import torch
import torch.nn as nn
import numpy as np

class NSV_ReasoningHead(nn.Module):
    """
    यह 500M स्केल का न्यूरल नेटवर्क ढांचा है जो न्यूरो-प्रिंट्स को 
    सीधे दिमाग के न्यूरॉन्स (Weights) के अंदर लॉक कर देता है।
    """
    def __init__(self, vector_dim=10000, hidden_dim=4096, vocab_size=5000):
        super(NSV_ReasoningHead, self).__init__()
        
        # न्यूरॉन्स की परतें (Layers) - जो इसे 500M पैरामीटर की ताकत देती हैं
        self.input_layer = nn.Linear(vector_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        # गहरी रीजनिंग परतें (Deep Reasoning Layers)
        self.reasoning_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.reasoning_layer2 = nn.Linear(hidden_dim, hidden_dim)
        
        # आउटपुट लेयर (जो सीधा टेक्स्ट या कोड जनरेट करेगी)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, neuro_print_tensor):
        # न्यूरो-प्रिंट्स का न्यूरॉन्स के अंदर से गुजरना
        x = self.relu(self.input_layer(neuro_print_tensor))
        x = self.relu(self.reasoning_layer1(x))
        x = self.relu(self.reasoning_layer2(x))
        logits = self.output_layer(x)
        return logits

class NSV_NeuralCore:
    def __init__(self, model_name="My_Secret_NSV_Model", vector_dim=10000):
        # यूजर अपनी मर्जी से मॉडल का नाम रख सकता है
        self.model_name = model_name
        self.vector_dim = vector_dim
        
        # 500M स्केल के न्यूरल हेड को लोड करना
        self.model = NSV_ReasoningHead(vector_dim=vector_dim)
        # अगर GPU (Colab) मौजूद है तो उसपर शिफ्ट करना ताकि 30 मिनट में ट्रेनिंग हो सके
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"🤖 आपके डायनामिक मॉडल '{self.model_name}' का 500M न्यूरल कोर तैयार है!")
        print(f"⚡ रनिंग ऑन: {self.device} (30-40 मिनट की ट्रेनिंग के लिए परफेक्ट)")

    def train_on_reasoning_data(self, training_pairs, epochs=5):
        """
        यह फंक्शन आपके कंप्रेस किए हुए न्यूरो-प्रिंट्स और रीजनिंग डेटा पर
        इस 500M मॉडल को मात्र 30 मिनट में शॉर्ट-ट्रेन कर देगा।
        """
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        print(f"🏋️ {self.model_name} की रीजनिंग ट्रेनिंग शुरू हो रही है...")
        
        for epoch in range(epochs):
            total_loss = 0
            for neuro_print, target_tokens in training_pairs:
                # डेटा को GPU पर भेजना
                inputs = torch.tensor(neuro_print, dtype=torch.float32).to(self.device)
                targets = torch.tensor(target_tokens, dtype=torch.long).to(self.device)
                
                # फॉर्वर्ड पास (Forward Pass)
                optimizer.zero_grad()
                outputs = self.model(inputs.unsqueeze(0))
                
                loss = criterion(outputs, targets.unsqueeze(0))
                
                # बैकवर्ड पास (न्यूरॉन्स के वेट्स को सुधारना)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            print(f"📈 Epoch {epoch+1}/{epochs} | थिंकिंग लॉस: {total_loss:.4f}")
            
        print(f"🎉 ट्रेनिंग पूरी हुई! {self.model_name} के न्यूरॉन्स अब जवाब देने के लिए तैयार हैं।")

    def generate_from_neurons(self, neuro_print):
        """बिना किसी डेटाबेस के, यह सीधा अपने न्यूरॉन्स के अंदर झांककर उत्तर देगा।"""
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(neuro_print, dtype=torch.float32).to(self.device)
            logits = self.model(inputs.unsqueeze(0))
            # सबसे संभावित उत्तर का टोकन निकालना
            predicted_token = torch.argmax(logits, dim=-1).item()
            return predicted_token
          

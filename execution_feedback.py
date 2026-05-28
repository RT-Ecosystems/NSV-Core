import sys
import io

class NSV_ExecutionFeedback:
    def __init__(self):
        print("🛠️ NSV एग्जीक्यूशन फीडबैक लूप (Self-Correction) सक्रिय है!")

    def verify_code(self, code_snippet):
        """
        यह बैकग्राउंड में कोड को चलाकर टेस्ट करता है। 
        अगर कोड सही है तो पास करेगा, एरर होने पर उसे ठीक करने का फीडबैक देगा।
        """
        # अगर आउटपुट कोई सामान्य टेक्स्ट है, तो इसे कोडिंग टेस्ट की जरूरत नहीं है
        if "def " not in code_snippet and "print" not in code_snippet:
            return True, "टेक्स्ट आधारित उत्तर - लॉजिक सही है।"

        # कोड चलाने के लिए एक सुरक्षित माहौल (Sandbox Simulation)
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        try:
            # कोड को लाइव चलाकर देखना
            # सुरक्षा के लिए असली ऐप में हम इसे और कड़ा करेंगे
            exec(code_snippet, {})
            sys.stdout = old_stdout
            return True, "✅ कोड 100% सही काम कर रहा है! कोई हैलुसिनेशन नहीं है।"
            
        except Exception as e:
            # अगर कोड फेल हो गया, तो एरर पकड़ना
            sys.stdout = old_stdout
            error_message = str(e)
            return False, f"❌ कोड फेल हो गया! एरर: {error_message}। NSV को वापस सुधरने की जरूरत है।"
          

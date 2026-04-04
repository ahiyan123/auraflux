import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .settings import settings

class AurafluxSupervisor:
    def __init__(self):
        # Load local Gemma 4 with 4-bit quantization for 24GB VRAM cards
        self.tokenizer = AutoTokenizer.from_pretrained(settings.SUPERVISOR)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.SUPERVISOR,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.bfloat16
        )

    def judge(self, user_prompt, logic_out, audit_out):
        # Utilizing Gemma 4's Native Thinking Mode
        input_text = f"""<|think|>
        Analyze the logic from Qwen: {logic_out}
        Analyze the audit from GPT-OSS: {audit_out}
        Original Intent: {user_prompt}
        
        Identify gaps and synthesize the final 'Auraflux' consensus.
        <channel|>"""
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=1024)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Singleton for the app
brain = AurafluxSupervisor()

import os

class Settings:
    # No hardcoded keys. Everything pulled from system environment.
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # 2026 Model Registry
    SUPERVISOR = "google/gemma-4-31b-it"        # Dense 31B
    LOGIC_AGENT = "Qwen/Qwen3.5-397B-A17B-Instruct" # 17B Active MoE
    AUDIT_AGENT = "openai/gpt-oss-120b"         # Released Aug 2025

settings = Settings()

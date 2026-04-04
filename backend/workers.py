from huggingface_hub import AsyncInferenceClient
from .settings import settings

client = AsyncInferenceClient(token=settings.HF_TOKEN)

async def call_logic_expert(prompt):
    """Qwen 3.5 handles complex engineering and reasoning."""
    response = await client.text_generation(
        prompt, model=settings.LOGIC_AGENT, max_new_tokens=512
    )
    return response

async def call_audit_expert(prompt):
    """GPT-OSS handles syntax, grammar, and rapid auditing."""
    response = await client.text_generation(
        prompt, model=settings.AUDIT_AGENT, max_new_tokens=300
    )
    return response

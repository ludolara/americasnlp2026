"""Shared constants for training data formatting."""

AYA_END_OF_TURN_TOKEN = "<|END_OF_TURN_TOKEN|>"
AYA_REQUIRED_CHAT_TOKENS = (
    "<|START_OF_TURN_TOKEN|>",
    AYA_END_OF_TURN_TOKEN,
    "<|USER_TOKEN|>",
    "<|CHATBOT_TOKEN|>",
    "<|SYSTEM_TOKEN|>",
)
AYA_BASE_CHAT_TEMPLATE = """{{ bos_token }}{% for message in messages %}{% set role = message['role']|lower %}<|START_OF_TURN_TOKEN|>{% if role == 'user' %}<|USER_TOKEN|>{{ message['content'] }}{% elif role == 'assistant' or role == 'chatbot' %}<|CHATBOT_TOKEN|>{{ message['content'] }}{% elif role == 'system' %}<|SYSTEM_TOKEN|>{{ message['content'] }}{% else %}{{- raise_exception("Unsupported role: " ~ message['role']) -}}{% endif %}<|END_OF_TURN_TOKEN|>{% endfor %}{% if add_generation_prompt %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{% endif %}"""
SPANISH_LANGUAGE_NAME = "español"


# config.py for Patient Metrics Extraction Agent

import os
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()

class ConfigError(Exception):
    pass

class Config:
    # API Keys and Tokens
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    METRICS_VALIDATOR_TOKEN = os.getenv("METRICS_VALIDATOR_TOKEN")
    EHR_API_TOKEN = os.getenv("EHR_API_TOKEN")
    AUDIT_LOGGER_TOKEN = os.getenv("AUDIT_LOGGER_TOKEN")

    # API URLs
    METRICS_VALIDATOR_URL = os.getenv("METRICS_VALIDATOR_URL")
    EHR_API_URL = os.getenv("EHR_API_URL")
    AUDIT_LOGGER_URL = os.getenv("AUDIT_LOGGER_URL")

    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
    LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "gpt-3.5-turbo")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1200"))
    LLM_SYSTEM_PROMPT = (
        "You are a professional healthcare data extraction agent. Your task is to extract, validate, and summarize key clinical metrics from patient medical reports. "
        "Always maintain patient confidentiality, adhere to compliance standards, and do not provide medical advice or diagnosis."
    )
    LLM_USER_PROMPT_TEMPLATE = (
        "Please upload or paste the patient medical report. The agent will extract and summarize key metrics such as vital signs, lab results, and medication dosages."
    )
    LLM_FEW_SHOT_EXAMPLES = [
        "Patient: John Doe, Age: 54. BP: 130/85 mmHg, HR: 78 bpm, WBC: 6.2 x10^9/L. Medications: Lisinopril 10mg daily.",
        "Discharge Summary: Temp 37.2°C, Resp Rate 16/min, Cholesterol 180 mg/dL. Prescribed: Atorvastatin 20mg at night."
    ]

    # Domain-specific settings
    DOMAIN = "healthcare"
    AGENT_NAME = "Patient Metrics Extraction Agent"
    MAX_REPORT_LENGTH = int(os.getenv("MAX_REPORT_LENGTH", "50000"))
    COMPLIANCE = ["HIPAA", "HITECH Act", "Institutional Review Board (IRB) standards"]
    SECURITY = {
        "encryption": "AES-256",
        "auth": "OAuth 2.0 with MFA",
        "session_timeout": 900,
        "pii_redaction": True
    }
    PERFORMANCE = {
        "response_time": 1.5,
        "throughput": 50,
        "resource_usage": {"vCPUs": 4, "RAM_GB": 16}
    }
    SCALABILITY = {
        "horizontal_scaling": True,
        "stateless": True,
        "external_state": True,
        "rate_limiting": True
    }

    # API Rate Limits
    API_RATE_LIMITS = {
        "EHR_API": "100 requests/minute",
        "Metrics_Validator": "200 requests/minute",
        "Audit_Logger": "500 requests/minute",
        "OpenAI_API": "As per OpenAI subscription"
    }

    # Validation and error handling
    @classmethod
    def validate(cls):
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not cls.METRICS_VALIDATOR_TOKEN or not cls.METRICS_VALIDATOR_URL:
            missing.append("METRICS_VALIDATOR_TOKEN or METRICS_VALIDATOR_URL")
        if not cls.EHR_API_TOKEN or not cls.EHR_API_URL:
            missing.append("EHR_API_TOKEN or EHR_API_URL")
        if not cls.AUDIT_LOGGER_TOKEN or not cls.AUDIT_LOGGER_URL:
            missing.append("AUDIT_LOGGER_TOKEN or AUDIT_LOGGER_URL")
        if missing:
            logger.error(f"Missing required configuration(s): {', '.join(missing)}")
            raise ConfigError(f"Missing required configuration(s): {', '.join(missing)}")

    @classmethod
    def get_llm_config(cls):
        return {
            "provider": cls.LLM_PROVIDER,
            "model": cls.LLM_MODEL,
            "fallback_model": cls.LLM_FALLBACK_MODEL,
            "temperature": cls.LLM_TEMPERATURE,
            "max_tokens": cls.LLM_MAX_TOKENS,
            "system_prompt": cls.LLM_SYSTEM_PROMPT,
            "user_prompt_template": cls.LLM_USER_PROMPT_TEMPLATE,
            "few_shot_examples": cls.LLM_FEW_SHOT_EXAMPLES
        }

    @classmethod
    def get_api_config(cls):
        return {
            "EHR_API": {
                "url": cls.EHR_API_URL,
                "token": cls.EHR_API_TOKEN,
                "rate_limit": cls.API_RATE_LIMITS["EHR_API"],
                "auth": "OAuth 2.0 with MFA"
            },
            "Metrics_Validator": {
                "url": cls.METRICS_VALIDATOR_URL,
                "token": cls.METRICS_VALIDATOR_TOKEN,
                "rate_limit": cls.API_RATE_LIMITS["Metrics_Validator"],
                "auth": "OAuth 2.0"
            },
            "Audit_Logger": {
                "url": cls.AUDIT_LOGGER_URL,
                "token": cls.AUDIT_LOGGER_TOKEN,
                "rate_limit": cls.API_RATE_LIMITS["Audit_Logger"],
                "auth": "OAuth 2.0"
            },
            "OpenAI_API": {
                "api_key": cls.OPENAI_API_KEY,
                "rate_limit": cls.API_RATE_LIMITS["OpenAI_API"],
                "auth": "API Key"
            }
        }

# Validate configuration at import
try:
    Config.validate()
except ConfigError as e:
    # Commented out to avoid codefence: raise
    # raise e
    logger.error(f"Configuration error: {e}")

# Default values and fallbacks are set above using os.getenv defaults.
# All error handling is centralized in Config.validate and logger.

# Usage example (commented out):
# llm_config = Config.get_llm_config()
# api_config = Config.get_api_config()

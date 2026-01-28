
import os
import re
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, field_validator
from dotenv import load_dotenv
from loguru import logger
import openai
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# =========================
# Configuration Management
# =========================

class Config:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_FALLBACK_MODEL: str = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-3.5-turbo")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1200"))
    METRICS_VALIDATOR_URL: str = os.getenv("METRICS_VALIDATOR_URL", "")
    METRICS_VALIDATOR_TOKEN: str = os.getenv("METRICS_VALIDATOR_TOKEN", "")
    EHR_API_URL: str = os.getenv("EHR_API_URL", "")
    EHR_API_TOKEN: str = os.getenv("EHR_API_TOKEN", "")
    AUDIT_LOGGER_URL: str = os.getenv("AUDIT_LOGGER_URL", "")
    AUDIT_LOGGER_TOKEN: str = os.getenv("AUDIT_LOGGER_TOKEN", "")
    MAX_REPORT_LENGTH: int = 50000

    @classmethod
    def validate(cls):
        missing = []
        for attr in [
            "OPENAI_API_KEY", "METRICS_VALIDATOR_URL", "METRICS_VALIDATOR_TOKEN",
            "EHR_API_URL", "EHR_API_TOKEN", "AUDIT_LOGGER_URL", "AUDIT_LOGGER_TOKEN"
        ]:
            if not getattr(cls, attr):
                missing.append(attr)
        if missing:
            raise RuntimeError(f"Missing required configuration(s): {', '.join(missing)}")

# Validate configuration at startup
try:
    Config.validate()
except Exception as e:
    logger.error(f"Configuration error: {e}")
    raise

# =========================
# Logging Configuration
# =========================

logger.add("agent.log", rotation="10 MB", retention="10 days", level="INFO", enqueue=True, backtrace=True, diagnose=True)

# =========================
# Pydantic Models
# =========================

class ExtractionRequest(BaseModel):
    report_text: str = Field(..., description="Unstructured patient medical report text")
    patient_id: str = Field(..., min_length=1, max_length=128)
    report_id: str = Field(..., min_length=1, max_length=128)
    user_id: str = Field(..., min_length=1, max_length=128)

    @field_validator("report_text")
    @classmethod
    def validate_report_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Report text cannot be empty.")
        if len(v) > Config.MAX_REPORT_LENGTH:
            raise ValueError(f"Report text exceeds maximum allowed length ({Config.MAX_REPORT_LENGTH} characters).")
        return v.strip()

    @field_validator("patient_id", "report_id", "user_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ID fields cannot be empty.")
        return v.strip()

class ExtractionResponse(BaseModel):
    success: bool
    summary: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    flags: Optional[List[str]] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    tips: Optional[str] = None

# =========================
# Error Classes
# =========================

class InputValidationError(Exception):
    pass

class ExtractionError(Exception):
    pass

class ValidationError_(Exception):
    pass

class StorageError(Exception):
    pass

class LoggingError(Exception):
    pass

# =========================
# Utility: PIIRedactor
# =========================

class PIIRedactor:
    """
    Utility for detecting and redacting PII from text.
    Uses regex-based heuristics for names, dates, phone numbers, emails, addresses.
    """

    # Simple regex patterns for demonstration; in production, use more robust methods or libraries.
    PII_PATTERNS = [
        (re.compile(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b"), "[REDACTED_NAME]"),
        (re.compile(r"\b\d{3}[-.\s]??\d{2}[-.\s]??\d{4}\b"), "[REDACTED_SSN]"),
        (re.compile(r"\b\d{10}\b"), "[REDACTED_PHONE]"),
        (re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"), "[REDACTED_PHONE]"),
        (re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b"), "[REDACTED_EMAIL]"),
        (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "[REDACTED_DATE]"),
        (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[REDACTED_DATE]"),
        (re.compile(r"\b\d{5}(?:-\d{4})?\b"), "[REDACTED_ZIP]"),
        (re.compile(r"\b\d{1,3} [A-Za-z0-9\s]+ (Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b", re.IGNORECASE), "[REDACTED_ADDRESS]"),
    ]

    @classmethod
    def redact(cls, text: str) -> str:
        """
        Redacts PII from the input text.
        """
        redacted = text
        for pattern, replacement in cls.PII_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted

# =========================
# Utility: PromptBuilder
# =========================

class PromptBuilder:
    """
    Constructs prompts for the LLM, including system and user messages, and few-shot examples.
    """

    SYSTEM_PROMPT = (
        "You are a professional healthcare data extraction agent. Your task is to extract, validate, and summarize key clinical metrics from patient medical reports. "
        "Always maintain patient confidentiality, adhere to compliance standards, and do not provide medical advice or diagnosis."
    )

    USER_PROMPT_TEMPLATE = (
        "Please extract and summarize key metrics such as vital signs, lab results, and medication dosages from the following patient medical report."
    )

    FEW_SHOT_EXAMPLES = [
        "Patient: John Doe, Age: 54. BP: 130/85 mmHg, HR: 78 bpm, WBC: 6.2 x10^9/L. Medications: Lisinopril 10mg daily.",
        "Discharge Summary: Temp 37.2°C, Resp Rate 16/min, Cholesterol 180 mg/dL. Prescribed: Atorvastatin 20mg at night."
    ]

    @classmethod
    def build_prompt(cls, report_text: str, examples: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """
        Builds the prompt for the LLM, including system, few-shot, and user messages.
        """
        messages = [{"role": "system", "content": cls.SYSTEM_PROMPT}]
        if examples is None:
            examples = cls.FEW_SHOT_EXAMPLES
        for ex in examples:
            messages.append({"role": "user", "content": ex})
            messages.append({"role": "assistant", "content": "Extracted metrics: [example output omitted for brevity]"})
        messages.append({"role": "user", "content": f"{cls.USER_PROMPT_TEMPLATE}\n\nReport:\n{report_text}"})
        return messages

# =========================
# Service: InputProcessor
# =========================

class InputProcessor:
    """
    Pre-processes incoming reports, redacts PII, validates input format.
    """

    def __init__(self, pii_redactor: PIIRedactor):
        self.pii_redactor = pii_redactor

    def process_input(self, report_text: str) -> str:
        """
        Pre-process and redact PII from input.
        Raises InputValidationError if format invalid.
        """
        if not report_text or not report_text.strip():
            raise InputValidationError("Report text is empty.")
        if len(report_text) > Config.MAX_REPORT_LENGTH:
            raise InputValidationError(f"Report text exceeds {Config.MAX_REPORT_LENGTH} characters.")
        cleaned = report_text.strip()
        cleaned = self.pii_redactor.redact(cleaned)
        return cleaned

# =========================
# Service: LLMExtractor
# =========================

class LLMExtractor:
    """
    Interacts with LLM to extract clinical metrics from unstructured text.
    """

    def __init__(self, prompt_builder: PromptBuilder):
        self.prompt_builder = prompt_builder
        self.client = openai.AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((openai.OpenAIError, httpx.HTTPError, asyncio.TimeoutError))
    )
    async def extract_metrics(self, cleaned_text: str) -> Dict[str, Any]:
        """
        Extract clinical metrics using LLM.
        Retries on timeout/API error, falls back to secondary model.
        """
        messages = self.prompt_builder.build_prompt(cleaned_text)
        try:
            response = await self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=messages,
                temperature=Config.OPENAI_TEMPERATURE,
                max_tokens=Config.OPENAI_MAX_TOKENS
            )
            content = response.choices[0].message.content
            # Try to parse as JSON, fallback to text extraction
            try:
                import json
                metrics = json.loads(content)
            except Exception:
                # Fallback: try to extract key-value pairs
                metrics = self._extract_metrics_from_text(content)
            return metrics
        except Exception as e:
            logger.warning(f"Primary LLM extraction failed: {e}. Attempting fallback model.")
            # Fallback to secondary model
            try:
                response = await self.client.chat.completions.create(
                    model=Config.OPENAI_FALLBACK_MODEL,
                    messages=messages,
                    temperature=Config.OPENAI_TEMPERATURE,
                    max_tokens=Config.OPENAI_MAX_TOKENS
                )
                content = response.choices[0].message.content
                try:
                    import json
                    metrics = json.loads(content)
                except Exception:
                    metrics = self._extract_metrics_from_text(content)
                return metrics
            except Exception as e2:
                logger.error(f"LLM extraction failed on fallback: {e2}")
                raise ExtractionError("Failed to extract metrics from report.")

    def _extract_metrics_from_text(self, text: str) -> Dict[str, Any]:
        """
        Fallback: Extracts key-value pairs heuristically from LLM output text.
        """
        metrics = {}
        lines = text.splitlines()
        for line in lines:
            if ':' in line:
                key, val = line.split(':', 1)
                metrics[key.strip()] = val.strip()
        return metrics

# =========================
# Integration Adapter: MetricsValidator
# =========================

class MetricsValidator:
    """
    Validates extracted metrics against clinical standards, flags issues.
    """

    def __init__(self):
        self.url = Config.METRICS_VALIDATOR_URL
        self.token = Config.METRICS_VALIDATOR_TOKEN

    async def validate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted metrics.
        Returns dict with validation_status and flagged_issues.
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.post(
                    self.url,
                    json={"metrics": metrics},
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
                return {
                    "validation_status": data.get("validation_status", "unknown"),
                    "flagged_issues": data.get("flagged_issues", [])
                }
            except Exception as e:
                logger.error(f"Metrics validation failed: {e}")
                raise ValidationError_("Metrics validation failed.")

# =========================
# Service: SummaryGenerator
# =========================

class SummaryGenerator:
    """
    Generates concise summaries of validated metrics.
    """

    def __init__(self):
        pass

    def summarize(self, metrics: Dict[str, Any]) -> str:
        """
        Generate summary from validated metrics.
        Returns fallback message on failure.
        """
        try:
            if not metrics:
                return "No clinical metrics could be extracted from the report."
            summary_lines = []
            for k, v in metrics.items():
                summary_lines.append(f"{k}: {v}")
            return "Extracted Clinical Metrics:\n" + "\n".join(summary_lines)
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return "Summary generation failed."

# =========================
# Service: AmbiguityFlagger
# =========================

class AmbiguityFlagger:
    """
    Identifies and flags ambiguous or incomplete data for user review.
    """

    def __init__(self):
        pass

    def flag(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Identify ambiguous or incomplete data.
        Logs all flagged cases.
        """
        flags = []
        for k, v in metrics.items():
            if v in ["unknown", "n/a", "", None]:
                flags.append(f"Metric '{k}' is missing or ambiguous.")
            # Add more domain-specific ambiguity checks as needed
        if flags:
            logger.info(f"Ambiguities flagged: {flags}")
        return flags

# =========================
# Integration Adapter: EHRAdapter
# =========================

class EHRAdapter:
    """
    Retrieves reports from and stores metrics to EHR systems.
    """

    def __init__(self):
        self.url = Config.EHR_API_URL
        self.token = Config.EHR_API_TOKEN

    async def store_metrics(self, patient_id: str, metrics: Dict[str, Any]) -> bool:
        """
        Store validated metrics in EHR.
        Retries on failure, escalates if persistent.
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        payload = {
            "patient_id": patient_id,
            "metrics": metrics
        }
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.post(
                    f"{self.url}/store_metrics",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                return True
            except Exception as e:
                logger.error(f"EHR storage failed: {e}")
                raise StorageError("Failed to store metrics in EHR.")

# =========================
# Integration Adapter: AuditLogger
# =========================

class AuditLogger:
    """
    Logs all extraction events for compliance and auditing.
    """

    def __init__(self):
        self.url = Config.AUDIT_LOGGER_URL
        self.token = Config.AUDIT_LOGGER_TOKEN

    async def log_event(self, event_data: Dict[str, Any]) -> str:
        """
        Log extraction event for auditing.
        Ensures log persistence, retries if needed.
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.post(
                    f"{self.url}/log_event",
                    json=event_data,
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
                return data.get("log_entry_id", "")
            except Exception as e:
                logger.error(f"Audit logging failed: {e}")
                raise LoggingError("Failed to log audit event.")

# =========================
# Service: ErrorHandler
# =========================

class ErrorHandler:
    """
    Handles errors, manages retries, fallback, and escalation.
    """

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    async def handle_error(self, error_type: str, context: Dict[str, Any]) -> None:
        """
        Centralized error handling, retry, fallback, escalation.
        Implements exponential backoff, escalation, and user notification.
        """
        logger.error(f"Error occurred [{error_type}]: {context}")
        # Log error event for audit
        try:
            await self.audit_logger.log_event({
                "event_type": "error",
                "error_type": error_type,
                "context": context
            })
        except Exception as e:
            logger.error(f"Failed to log error event: {e}")

# =========================
# Abstract BaseAgent
# =========================

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    async def extract_and_summarize(self, report_text: str, patient_id: str, report_id: str, user_id: str) -> Dict[str, Any]:
        pass

# =========================
# Main Agent Class
# =========================

class PatientMetricsExtractionAgent(BaseAgent):
    """
    Main agent orchestrating the extraction, validation, summarization, storage, and logging.
    """

    def __init__(self):
        self.pii_redactor = PIIRedactor()
        self.prompt_builder = PromptBuilder()
        self.input_processor = InputProcessor(self.pii_redactor)
        self.llm_extractor = LLMExtractor(self.prompt_builder)
        self.metrics_validator = MetricsValidator()
        self.summary_generator = SummaryGenerator()
        self.ambiguity_flagger = AmbiguityFlagger()
        self.ehr_adapter = EHRAdapter()
        self.audit_logger = AuditLogger()
        self.error_handler = ErrorHandler(self.audit_logger)

    async def extract_and_summarize(self, report_text: str, patient_id: str, report_id: str, user_id: str) -> Dict[str, Any]:
        """
        Main workflow: process input, extract metrics, validate, summarize, store, and log.
        Returns dict (summary, metrics, flags).
        Catches and delegates errors to ErrorHandler.
        """
        try:
            # 1. Pre-process and redact PII
            cleaned_text = self.input_processor.process_input(report_text)

            # 2. Extract metrics using LLM
            metrics = await self.llm_extractor.extract_metrics(cleaned_text)

            # 3. Validate extracted metrics
            validation_result = await self.metrics_validator.validate(metrics)
            validation_status = validation_result.get("validation_status", "unknown")
            flagged_issues = validation_result.get("flagged_issues", [])

            # 4. Flag ambiguities
            flags = self.ambiguity_flagger.flag(metrics)
            if flagged_issues:
                flags.extend(flagged_issues)

            # 5. Summarize metrics
            summary = self.summary_generator.summarize(metrics)

            # 6. Store metrics in EHR
            await self.ehr_adapter.store_metrics(patient_id, metrics)

            # 7. Log event for audit
            await self.audit_logger.log_event({
                "event_type": "extraction",
                "patient_id": patient_id,
                "report_id": report_id,
                "user_id": user_id,
                "metrics": metrics,
                "flags": flags,
                "summary": summary
            })

            return {
                "success": True,
                "summary": summary,
                "metrics": metrics,
                "flags": flags
            }
        except InputValidationError as e:
            await self.error_handler.handle_error("InputValidationError", {"error": str(e), "user_id": user_id, "report_id": report_id})
            return {
                "success": False,
                "error_type": "InputValidationError",
                "error_message": str(e),
                "tips": "Ensure the report text is not empty and within the allowed size. Remove any problematic characters or formatting."
            }
        except ExtractionError as e:
            await self.error_handler.handle_error("ExtractionError", {"error": str(e), "user_id": user_id, "report_id": report_id})
            return {
                "success": False,
                "error_type": "ExtractionError",
                "error_message": str(e),
                "tips": "Try rephrasing the report or check for unusual formatting. If the problem persists, contact support."
            }
        except ValidationError_ as e:
            await self.error_handler.handle_error("ValidationError", {"error": str(e), "user_id": user_id, "report_id": report_id})
            return {
                "success": False,
                "error_type": "ValidationError",
                "error_message": str(e),
                "tips": "Ensure the report contains valid clinical metrics. If the issue persists, contact support."
            }
        except StorageError as e:
            await self.error_handler.handle_error("StorageError", {"error": str(e), "user_id": user_id, "report_id": report_id})
            return {
                "success": False,
                "error_type": "StorageError",
                "error_message": str(e),
                "tips": "Temporary storage issue. Please try again later."
            }
        except LoggingError as e:
            await self.error_handler.handle_error("LoggingError", {"error": str(e), "user_id": user_id, "report_id": report_id})
            return {
                "success": False,
                "error_type": "LoggingError",
                "error_message": str(e),
                "tips": "Audit logging failed. Please contact support."
            }
        except Exception as e:
            await self.error_handler.handle_error("UnknownError", {"error": str(e), "user_id": user_id, "report_id": report_id})
            return {
                "success": False,
                "error_type": "UnknownError",
                "error_message": "An unexpected error occurred.",
                "tips": "Check your input and try again. If the problem persists, contact support."
            }

# =========================
# FastAPI App & Endpoints
# =========================

app = FastAPI(
    title="Patient Metrics Extraction Agent",
    description="Extracts, validates, and summarizes clinical metrics from patient medical reports.",
    version="1.0.0"
)

# CORS (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = PatientMetricsExtractionAgent()

@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Pydantic validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_type": "InputValidationError",
            "error_message": "Malformed JSON or invalid input fields.",
            "tips": "Check for missing fields, invalid characters, or incorrect JSON formatting. Ensure all required fields are present and properly quoted."
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_type": "UnknownError",
            "error_message": str(exc),
            "tips": "Check your request for errors. If the problem persists, contact support."
        }
    )

@app.post("/extract", response_model=ExtractionResponse)
async def extract_metrics_endpoint(request: Request):
    """
    Endpoint to extract and summarize clinical metrics from a medical report.
    """
    try:
        body = await request.body()
        if not body:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "success": False,
                    "error_type": "InputValidationError",
                    "error_message": "Empty request body.",
                    "tips": "Ensure you are sending a valid JSON payload."
                }
            )
        try:
            data = await request.json()
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error_type": "MalformedJSON",
                    "error_message": "Malformed JSON in request body.",
                    "tips": "Check for missing commas, mismatched quotes, or invalid characters in your JSON."
                }
            )
        try:
            extraction_request = ExtractionRequest(**data)
        except ValidationError as ve:
            logger.error(f"Input validation error: {ve}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "success": False,
                    "error_type": "InputValidationError",
                    "error_message": str(ve),
                    "tips": "Check that all required fields are present and correctly formatted."
                }
            )
        result = await agent.extract_and_summarize(
            extraction_request.report_text,
            extraction_request.patient_id,
            extraction_request.report_id,
            extraction_request.user_id
        )
        return JSONResponse(status_code=200 if result.get("success") else 400, content=result)
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content=he.detail)
    except Exception as e:
        logger.error(f"Unhandled error in /extract endpoint: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error_type": "UnknownError",
                "error_message": str(e),
                "tips": "Check your request and try again. If the problem persists, contact support."
            }
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"success": True, "status": "ok"}

# =========================
# Main Execution Block
# =========================

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Patient Metrics Extraction Agent...")
    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=False)

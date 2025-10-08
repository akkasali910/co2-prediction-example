import re

class Guardrails:
    """A simple class to enforce input and output guardrails for the RAG application."""
    def __init__(self):
        # 1. Denied topics/keywords filter
        self.denied_keywords = [
            "politics", "religion", "violence", "hate speech", "adult content",
            # --- Add your new denied topics here ---
            "finance",
            "medical advice",
        ]

        # 2. PII detection using regular expressions for output sanitization
        self.pii_patterns = {
            "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "PHONE": re.compile(r'\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            # Example for a simple IP address regex. Add more as needed.
            "IP_ADDRESS": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            # --- Add your new PII patterns here ---
            "CREDIT_CARD": re.compile(r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b'),
        }

    def check_input(self, query: str) -> (bool, str):
        """Checks user input against guardrails. Returns (is_safe, message)."""
        lower_query = query.lower()
        for keyword in self.denied_keywords:
            if keyword in lower_query:
                return False, f"I cannot answer questions about '{keyword}'. Please stick to topics related to the indexed files."
        return True, ""

    def sanitize_output(self, text: str) -> str:
        """Sanitizes the output by redacting PII."""
        sanitized_text = text
        for pii_type, pattern in self.pii_patterns.items():
            sanitized_text = pattern.sub(f"[{pii_type}_REDACTED]", sanitized_text)
        return sanitized_text

    def stream_and_sanitize(self, streaming_response):
        """A generator that sanitizes tokens as they are streamed."""
        for token in streaming_response.response_gen:
            yield self.sanitize_output(token)
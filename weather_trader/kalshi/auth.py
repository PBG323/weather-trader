"""
Kalshi RSA key-pair authentication.

Signs API requests using RSA-PSS with the user's private key.
"""

import base64
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class KalshiAuth:
    """Handles RSA-PSS signing for Kalshi API authentication."""

    def __init__(self, key_id: str = "", private_key_path: str = ""):
        from ..config import config
        self._key_id = key_id or config.kalshi.key_id
        self._private_key_path = private_key_path or config.kalshi.private_key_path
        self._private_key: rsa.RSAPrivateKey | None = None

        if self._private_key_path and Path(self._private_key_path).exists():
            self._load_private_key()

    def _load_private_key(self):
        """Load RSA private key from PEM file."""
        pem_data = Path(self._private_key_path).read_bytes()
        self._private_key = serialization.load_pem_private_key(pem_data, password=None)

    @property
    def is_configured(self) -> bool:
        """Check if API credentials are configured."""
        return bool(self._key_id and self._private_key is not None)

    @property
    def key_id(self) -> str:
        return self._key_id

    def validate_for_trading(self) -> tuple[bool, str]:
        """Validate that credentials are ready for live trading."""
        if not self._key_id:
            return False, "KALSHI_KEY_ID not set in environment"
        if not self._private_key_path:
            return False, "KALSHI_PRIVATE_KEY_PATH not set in environment"
        if not Path(self._private_key_path).exists():
            return False, f"Private key file not found: {self._private_key_path}"
        if self._private_key is None:
            return False, "Failed to load private key from PEM file"
        return True, "Kalshi credentials validated"

    def get_auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Generate signed authentication headers for a Kalshi API request.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            path: API path without base URL (e.g., /trade-api/v2/portfolio/balance)

        Returns:
            Dict of authentication headers.
        """
        if not self.is_configured:
            raise RuntimeError("Kalshi auth not configured. Set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH.")

        timestamp_ms = str(int(time.time() * 1000))
        message = timestamp_ms + method.upper() + path
        signature = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "Content-Type": "application/json",
        }

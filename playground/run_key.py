"""Decrypt the OpenRouter key a researcher supplied for their own playground run.

HumanStudy-Hub never stores a researcher's key in plaintext. The web app encrypts
it with AES-256-GCM under a secret shared with this workflow
(`PLAYGROUND_KEY_SECRET`), so the private jobs repository only ever holds
ciphertext. This module reverses that on the runner.
"""

import base64
import hashlib
import os
from typing import Any, Mapping, Optional


def _shared_secret() -> Optional[bytes]:
    secret = os.environ.get("PLAYGROUND_KEY_SECRET", "").strip()
    if not secret:
        return None
    # The web app derives the same 32-byte key with SHA-256 over the secret.
    return hashlib.sha256(secret.encode("utf-8")).digest()


def decrypt_api_key(sealed: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return the researcher's OpenRouter key, or None when the run has none.

    Raises ValueError when a key was supplied but cannot be opened, because
    silently falling back to the shared key would spend the project's budget on
    a run the researcher expected to pay for.
    """
    if not sealed:
        return None
    key = _shared_secret()
    if key is None:
        raise ValueError("PLAYGROUND_KEY_SECRET is not configured, so the supplied API key cannot be opened.")
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        iv = base64.b64decode(sealed["iv"])
        ciphertext = base64.b64decode(sealed["ciphertext"])
        tag = base64.b64decode(sealed["tag"])
        plaintext = AESGCM(key).decrypt(iv, ciphertext + tag, None)
    except KeyError as exc:
        raise ValueError(f"The sealed API key is missing the {exc} field.") from exc
    except Exception as exc:
        raise ValueError("The supplied API key could not be decrypted with the configured secret.") from exc
    opened = plaintext.decode("utf-8").strip()
    if not opened:
        raise ValueError("The supplied API key decrypted to an empty value.")
    return opened

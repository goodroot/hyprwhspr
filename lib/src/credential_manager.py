"""
Secure credential storage for API keys
Stores credentials with restricted file permissions (0600)
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

try:
    from .output_control import log_info, log_error, log_warning
except ImportError:
    from output_control import log_info, log_error, log_warning

try:
    from .paths import CREDENTIALS_DIR, CREDENTIALS_FILE
except ImportError:
    from paths import CREDENTIALS_DIR, CREDENTIALS_FILE

try:
    from .config_manager import expand_env
except ImportError:
    from config_manager import expand_env


def _ensure_credentials_dir():
    """Ensure credentials directory exists"""
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)


def _load_credentials() -> Dict[str, str]:
    """Load credentials from file"""
    if not CREDENTIALS_FILE.exists():
        return {}
    
    try:
        with open(CREDENTIALS_FILE, 'r', encoding='utf-8') as f:
            credentials = json.load(f)
        if not isinstance(credentials, dict) or not all(
            isinstance(provider, str) and isinstance(key, str)
            for provider, key in credentials.items()
        ):
            raise ValueError("credentials must be a string-to-string mapping")
        return credentials
    except (json.JSONDecodeError, OSError, ValueError) as e:
        log_warning(f"Error reading credentials file: {e}")
        return {}


def _save_credentials(credentials: Dict[str, str]):
    """Save credentials to file with restricted permissions"""
    _ensure_credentials_dir()
    
    temp_file = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{CREDENTIALS_FILE.name}.",
            suffix=".tmp",
            dir=CREDENTIALS_FILE.parent,
        )
        temp_file = Path(temp_name)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(credentials, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.chmod(temp_file, 0o600)

        os.replace(temp_file, CREDENTIALS_FILE)

    except Exception as e:
        log_error(f"Failed to save credentials: {e}")
        if temp_file is not None:
            try:
                temp_file.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def save_credential(provider: str, key: str) -> bool:
    """
    Save API key for a provider.
    
    Args:
        provider: Provider identifier (e.g., 'openai', 'groq')
        key: API key to store
    
    Returns:
        True if successful, False otherwise
    """
    try:
        credentials = _load_credentials()
        credentials[provider] = key
        _save_credentials(credentials)
        return True
    except Exception as e:
        log_error(f"Failed to save credential for {provider}: {e}")
        return False


def get_credential(provider: str) -> Optional[str]:
    """
    Retrieve API key for a provider. Supports ${VAR} expansion against env vars.

    Args:
        provider: Provider identifier

    Returns:
        API key if found, None otherwise
    """
    credentials = _load_credentials()
    value = credentials.get(provider)
    return expand_env(value) if isinstance(value, str) else value


def list_credentials() -> Dict[str, str]:
    """
    List all stored credentials (keys are masked).
    
    Returns:
        Dictionary mapping provider to masked key (e.g., 'sk-...****')
    """
    credentials = _load_credentials()
    # ${VAR} tokens are not expanded here — they appear as literals in masked output.
    masked = {}
    
    for provider, key in credentials.items():
        if len(key) > 8:
            # Show first 4 chars and last 4 chars, mask the middle
            masked_key = f"{key[:4]}...{key[-4:]}"
        else:
            masked_key = "****"
        masked[provider] = masked_key
    
    return masked


def delete_credential(provider: str) -> bool:
    """
    Delete stored credential for a provider.
    
    Args:
        provider: Provider identifier
    
    Returns:
        True if successful, False otherwise
    """
    try:
        credentials = _load_credentials()
        if provider in credentials:
            del credentials[provider]
            _save_credentials(credentials)
            return True
        return False
    except Exception as e:
        log_error(f"Failed to delete credential for {provider}: {e}")
        return False


def mask_api_key(key: str) -> str:
    """
    Mask an API key for display purposes.
    
    Args:
        key: API key to mask
    
    Returns:
        Masked key (e.g., 'sk-...****')
    """
    if not key or len(key) < 8:
        return "****"
    
    if len(key) <= 12:
        return f"{key[:4]}...****"
    
    return f"{key[:6]}...{key[-4:]}"

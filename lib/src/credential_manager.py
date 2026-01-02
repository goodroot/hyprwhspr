"""
Secure credential storage for API keys
Stores credentials with restricted file permissions (0600)
"""

import json
import os
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


def _ensure_credentials_dir():
    """Ensure credentials directory exists"""
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)


def _load_credentials() -> Dict[str, str]:
    """Load credentials from file"""
    if not CREDENTIALS_FILE.exists():
        return {}
    
    try:
        with open(CREDENTIALS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        log_warning(f"Error reading credentials file: {e}")
        return {}


def _save_credentials(credentials: Dict[str, str]):
    """Save credentials to file with restricted permissions"""
    _ensure_credentials_dir()
    
    # Write to temporary file first, then move (atomic operation)
    temp_file = CREDENTIALS_FILE.with_suffix('.tmp')
    
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(credentials, f, indent=2)
        
        # Set permissions before moving (0600 = owner read/write only)
        os.chmod(temp_file, 0o600)
        
        # Atomic move
        if CREDENTIALS_FILE.exists():
            CREDENTIALS_FILE.unlink()
        temp_file.replace(CREDENTIALS_FILE)
        
        # Ensure permissions are correct (in case replace didn't preserve them)
        os.chmod(CREDENTIALS_FILE, 0o600)
        
    except IOError as e:
        log_error(f"Failed to save credentials: {e}")
        if temp_file.exists():
            temp_file.unlink()
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
    Retrieve API key for a provider.
    
    Args:
        provider: Provider identifier
    
    Returns:
        API key if found, None otherwise
    """
    credentials = _load_credentials()
    return credentials.get(provider)


def list_credentials() -> Dict[str, str]:
    """
    List all stored credentials (keys are masked).
    
    Returns:
        Dictionary mapping provider to masked key (e.g., 'sk-...****')
    """
    credentials = _load_credentials()
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


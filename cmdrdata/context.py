"""
Context management for customer IDs and metadata
"""

import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional

# Thread-local storage for context
_context = threading.local()


def set_customer_context(customer_id: str) -> None:
    """
    Set the customer ID for the current thread context.
    
    Args:
        customer_id: Customer identifier to set
    """
    _context.customer_id = customer_id


def get_customer_context() -> Optional[str]:
    """
    Get the current customer ID from thread context.
    
    Returns:
        Current customer ID or None
    """
    return getattr(_context, "customer_id", None)


def clear_customer_context() -> None:
    """
    Clear the customer ID from thread context.
    """
    if hasattr(_context, "customer_id"):
        delattr(_context, "customer_id")


@contextmanager
def customer_context(customer_id: str):
    """
    Context manager for temporarily setting customer ID.
    
    Args:
        customer_id: Customer identifier to use within context
    
    Example:
        with customer_context("customer-123"):
            # All API calls here will use customer-123
            response = client.chat.completions.create(...)
    """
    previous = get_customer_context()
    set_customer_context(customer_id)
    try:
        yield
    finally:
        if previous is not None:
            set_customer_context(previous)
        else:
            clear_customer_context()


def set_metadata_context(metadata: Dict[str, Any]) -> None:
    """
    Set metadata for the current thread context.
    
    Args:
        metadata: Metadata dictionary to set
    """
    _context.metadata = metadata


def get_metadata_context() -> Dict[str, Any]:
    """
    Get the current metadata from thread context.
    
    Returns:
        Current metadata dictionary or empty dict
    """
    return getattr(_context, "metadata", {})


def clear_metadata_context() -> None:
    """
    Clear the metadata from thread context.
    """
    if hasattr(_context, "metadata"):
        delattr(_context, "metadata")


def update_metadata_context(metadata: Dict[str, Any]) -> None:
    """
    Update (merge) metadata in the current thread context.
    
    Args:
        metadata: Metadata to merge with existing context
    """
    current = get_metadata_context()
    current.update(metadata)
    set_metadata_context(current)


@contextmanager
def metadata_context(metadata: Dict[str, Any]):
    """
    Context manager for temporarily setting metadata.
    
    Args:
        metadata: Metadata to use within context
    
    Example:
        with metadata_context({"feature": "chat", "experiment": "v2"}):
            # All API calls here will include this metadata
            response = client.chat.completions.create(...)
    """
    previous = get_metadata_context().copy()
    set_metadata_context(metadata)
    try:
        yield
    finally:
        set_metadata_context(previous)
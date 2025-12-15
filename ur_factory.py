"""Factory helpers for constructing UR interfaces consistently across projects."""

from ur_interface import UrInterface


def create_ur_interface() -> UrInterface:
    """Return a new :class:`UrInterface` instance with default settings."""
    return UrInterface()


__all__ = ["create_ur_interface", "UrInterface"]

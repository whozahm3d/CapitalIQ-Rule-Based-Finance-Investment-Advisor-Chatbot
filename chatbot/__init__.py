"""
chatbot/__init__.py
-------------------
Package initializer for the CapitalIQ chatbot module.

Author  : Ali Ahmad
Project : DecodeLabs AI Project 1 — Rule-Based AI Chatbot
Batch   : 2026 | Powered by DecodeLabs

Exposes the main CapitalIQBot class at the top-level package for convenient imports:
    from chatbot import CapitalIQBot
"""

from chatbot.bot import CapitalIQBot  # noqa: F401 — re-exported intentionally

__all__ = ["CapitalIQBot"]
__version__ = "2.0.0"
__author__ = "Ali Ahmad"

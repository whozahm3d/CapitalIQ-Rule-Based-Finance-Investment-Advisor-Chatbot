# CapitalIQ

**Rule-Based Finance & Investment Advisor Chatbot**

[![Python](https://img.shields.io/badge/Python-3.10%2B-306998?logo=python&logoColor=white)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/Dependencies-Standard%20Library%20Only-2e7d32)]()
[![Tests](https://img.shields.io/badge/Tests-61%20Passing-2e7d32)]()
[![License](https://img.shields.io/badge/License-MIT-546e7a)]()
[![DecodeLabs](https://img.shields.io/badge/DecodeLabs-Batch%202026-1a3c5e)]()

> DecodeLabs AI Industrial Training Programme · Project 1 · Batch 2026
> Author: **Ali Ahmad**

---

## Overview

CapitalIQ is a production-quality, rule-based conversational AI system built as Project 1 of the DecodeLabs Artificial Intelligence Industrial Training Programme. It operates as a Finance & Investment Advisor, responding to queries across 16 intent categories including stocks, bonds, ETFs, mutual funds, cryptocurrency, inflation, compound interest, portfolio strategy, and IPOs.

The system is intentionally built without machine learning dependencies — demonstrating that a well-engineered deterministic architecture using the **Input-Process-Output (IPO) model** and **O(1) dictionary-based intent routing** produces a system that is both robust in production and fully auditable for regulated-industry use cases.

---

## Architecture

### IPO Pipeline

```
Raw Input
    |
    v
[ SANITIZE ]          .lower().strip() + regex normalization + tokenization
    |
    v
[ MATCH INTENT ]      O(1) synonym lookup -> keyword voting -> confidence gate
    |
    v
[ GENERATE RESPONSE ] personalize -> log to memory -> display
    |
    v
Output
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `sanitizer.py` | Input normalization pipeline — lowercase, strip, regex clean, tokenize |
| `matcher.py` | Phase 1 keyword matcher + Phase 2 confidence scorer |
| `knowledge_base.py` | `BASE_KNOWLEDGE` dict, `SYNONYM_MAP`, `KEYWORD_INTENT_MAP` |
| `memory.py` | Session history log + named entity slot store |
| `bot.py` | Orchestration layer — connects all modules, runs the main loop |

### Why Dictionary over If-Elif

An `if-elif` ladder performs a sequential scan — O(n) worst-case. Every new intent added increases lookup time linearly. A Python dictionary is a hash map — O(1) constant time regardless of how many intents exist. Adding the 100th intent costs the same as adding the 5th. For any production intent router, this is the only correct data structure.

---

## Repository Structure

```
decodelabs-ai-project1/
|
+-- chatbot/
|   +-- __init__.py           Package entry point
|   +-- knowledge_base.py     Intent dictionary, synonym map, keyword map
|   +-- sanitizer.py          sanitize(), is_empty(), tokenize()
|   +-- matcher.py            Phase 1 + Phase 2 matchers, confidence scorer
|   +-- memory.py             ConversationMemory class
|   +-- bot.py                CapitalIQBot orchestration class + main loop
|
+-- notebooks/
|   +-- project1_chatbot.ipynb    19-cell Jupyter Notebook
|
+-- tests/
|   +-- __init__.py
|   +-- test_chatbot.py       61 unit tests across all modules
|
+-- report/
|   +-- capitaliq_report.pdf  Project technical report (14 pages)
|   +-- capitaliq_report.tex  LaTeX source
|
+-- assets/
+-- .gitignore
+-- requirements.txt
+-- README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

The core chatbot has **zero third-party dependencies** — it runs on the Python Standard Library alone. Jupyter and pytest are listed in `requirements.txt` as optional extras for the notebook and test suite.

### Installation

```bash
git clone https://github.com/whozahm3d/decodelabs-ai-project1.git
cd decodelabs-ai-project1
pip install -r requirements.txt
```

### Running the Chatbot

```bash
# Recommended
python -m chatbot.bot

# Alternative
python chatbot/bot.py
```

### Running the Notebook

```bash
jupyter notebook notebooks/project1_chatbot.ipynb
```

### Running the Test Suite

```bash
python -m pytest tests/ -v
```

---

## Sample Session

```
==============================================================
  CapitalIQ  v2.0.0  --  Finance & Investment Advisory
  DecodeLabs AI Internship  |  Batch 2026  |  Ali Ahmad
==============================================================
  Topics: stocks · bonds · ETFs · crypto · SIPs · IPOs
  Type 'exit' or 'quit' to end the session.
==============================================================

  You: hello
  CapitalIQ: Hello! I'm CapitalIQ, your Finance & Investment Advisor.
             Ask me about stocks, bonds, mutual funds, ETFs, crypto,
             inflation, SIPs, IPOs, and more.

  You: my name is Ali
  CapitalIQ: Nice to meet you, Ali! ...

  You: what is compound interest
  CapitalIQ: Compound interest is interest earned on both your original
             principal AND accumulated interest from previous periods...
             [Intent: compound_interest | Confidence: 0.250]

  You: btc
  CapitalIQ: Cryptocurrency is a decentralised digital currency...
             [Intent: crypto | Confidence: 1.000]

  You: quit
  CapitalIQ: It was a pleasure advising you today. Goodbye!

  [Session transcript displayed]
```

---

## Knowledge Base

16 intent categories with 70+ synonym and alias mappings.

| Intent | Trigger Examples |
|---|---|
| `greeting` | hi, hello, hey, help, good morning |
| `stocks` | stock, share, equity, dividend, NYSE |
| `bonds` | bond, treasury, coupon, debenture, g-sec |
| `mutual_funds` | mutual fund, NAV, AUM, expense ratio |
| `etf` | ETF, index fund, passive investing |
| `crypto` | bitcoin, BTC, ethereum, blockchain, DeFi |
| `inflation` | inflation, CPI, purchasing power |
| `compound_interest` | compounding, time value of money |
| `diversification` | diversify, asset allocation, hedge |
| `risk` | volatility, drawdown, risk tolerance |
| `portfolio` | holdings, rebalancing, 60/40 |
| `budgeting` | budget, 50/30/20, money management |
| `emergency_fund` | emergency fund, safety net, savings |
| `sip` | SIP, systematic investment plan, rupee-cost averaging |
| `ipo` | IPO, DRHP, public listing |
| `goodbye` | exit, quit, bye, done |

---

## Two-Phase Implementation

### Phase 1 — Logic Engine

- Multi-keyword intent matching via `KEYWORD_INTENT_MAP`
- O(1) synonym resolution via `SYNONYM_MAP`
- Fallback response for unrecognized inputs
- Graceful handling of empty, whitespace-only, and special-character inputs

### Phase 2 — Production Upgrade

Phase 2 is a strict superset of Phase 1 — every Phase 1 feature works unchanged.

| Feature | Implementation |
|---|---|
| Confidence Scoring | Normalized [0.0, 1.0] score; configurable threshold (default 0.15) |
| Conversation Memory | Full session transcript + named entity slot store |
| Name Extraction | Regex detection of introduction patterns (`my name is`, `I am`, `call me`) |
| Personalization | Stored name used in periodic responses |
| Session Summary | Full history and statistics displayed at session end |

---

## Test Coverage

```bash
python -m pytest tests/ -v --tb=short
```

| Test Class | Module | Tests |
|---|---|---|
| `TestSanitizer` | `sanitizer.py` | 15 |
| `TestMatcher` | `matcher.py` | 17 |
| `TestConversationMemory` | `memory.py` | 16 |
| `TestCapitalIQBot` | `bot.py` | 13 |
| **Total** | | **61 / 61 passing** |

Edge cases explicitly covered: `None` input, empty string, whitespace-only, special characters, mixed case, long natural sentences, embedded exit commands, name extraction.

---

## Design Notes

### The White-Box Advantage

Every response in CapitalIQ is fully traceable:

```
"BITCOIN!" -> sanitize -> "bitcoin" -> SYNONYM_MAP -> "crypto" -> KNOWLEDGE_BASE -> response
```

No hidden weights, no probability distributions, no hallucination risk. In regulated industries — finance (SEBI, SEC), healthcare (FDA AI/ML guidance), insurance (GDPR Article 22) — this kind of auditability is a compliance requirement, not a preference.

### Connection to Production AI Guardrail Systems

In modern hybrid architectures, the rule-based layer sits above the LLM as a deterministic filter:

```
User Input
    |
    +-- Rule match? -- YES --> Instant Response  (zero latency, zero cost, zero hallucination)
    |
    +-- No match   -- NO  --> Pass to LLM        (flexible, but probabilistic)
```

CapitalIQ's `CONFIDENCE_THRESHOLD` is the direct analogue of the routing threshold used in frameworks like NVIDIA NeMo Guardrails and Meta Llama Guard. High-confidence, known intents are resolved by the rule layer instantly. Only genuinely novel or ambiguous queries escalate to the LLM — reducing both cost and hallucination surface area.

---

## Tech Stack

| Component | Details |
|---|---|
| Language | Python 3.10+ |
| Dependencies | Standard Library only (`re`, `datetime`, `unittest`) |
| Notebook | Jupyter Notebook — 19 cells |
| Testing | `unittest` + `pytest` — 61 tests |
| Architecture | IPO Model, O(1) Hash Map, Modular OOP, Separation of Concerns |

---

## DecodeLabs Specification Compliance

| Requirement | Implementation |
|---|---|
| Input Loop | `while True` in `CapitalIQBot.run()` (`bot.py`) with `try/except` for graceful interruption |
| Sanitization | `sanitize()` in `sanitizer.py` — `.lower().strip()` + regex pipeline |
| Knowledge Base | `BASE_KNOWLEDGE` dict — 16 intents, `.get()` method used exclusively |
| Fallback | `FALLBACK_RESPONSE` returned when no keywords match or confidence < threshold |
| Exit Strategy | `is_exit_command()` in `matcher.py` — `break` exits the main loop cleanly |

All 5 mandatory requirements satisfied.

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*DecodeLabs AI Industrial Training Programme · Batch 2026*

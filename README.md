# 💹 CapitalIQ — Rule-Based Finance & Investment Advisor Chatbot

> **DecodeLabs AI Internship · Project 1 · Batch 2026**
> Built by **Ali Ahmad**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Standard Library](https://img.shields.io/badge/Dependencies-Standard%20Library%20Only-green)]()
[![Tests](https://img.shields.io/badge/Tests-35%20Unit%20Tests-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

---

## 📌 Project Overview

**CapitalIQ** is a production-quality, rule-based AI chatbot built as Project 1 of the DecodeLabs Artificial Intelligence Industrial Training programme. It operates as a Finance & Investment Advisor — answering questions about stocks, bonds, mutual funds, ETFs, cryptocurrency, inflation, compound interest, diversification, risk management, portfolio construction, budgeting, emergency funds, SIPs, and IPOs.

The project is intentionally built without machine learning — demonstrating that a well-engineered deterministic system using the **IPO (Input-Process-Output) model** and **O(1) dictionary-based intent matching** can be both robust and portfolio-worthy.

---

## 🏗️ Architecture: The IPO Model

```
Raw Input ──► SANITIZE ──► MATCH INTENT ──► GENERATE RESPONSE ──► Output
              (Stage 1)      (Stage 2)          (Stage 3)
```

| Stage | Module | Responsibility |
|-------|--------|----------------|
| **Input** | `sanitizer.py` | Lowercase, strip, remove noise, tokenize |
| **Process** | `matcher.py` | O(1) synonym lookup → keyword scoring → confidence gating |
| **Output** | `bot.py` | Personalize, log to memory, display response |

**Why dictionary over if-elif?**
An `if-elif` ladder is O(n) — lookup time grows linearly with the number of rules. A Python dictionary (hash map) is O(1) — lookup time is constant regardless of how many intents exist. Adding the 100th intent to a dict costs the same as adding the 5th. This is the correct data structure for any production intent router.

---

## 📁 Repository Structure

```
decodelabs-ai-project1/
│
├── chatbot/
│   ├── __init__.py          # Package initializer — exposes CapitalIQBot
│   ├── knowledge_base.py    # BASE_KNOWLEDGE, SYNONYM_MAP, KEYWORD_INTENT_MAP
│   ├── sanitizer.py         # sanitize(), is_empty(), tokenize()
│   ├── matcher.py           # P1 keyword matcher + P2 confidence scorer
│   ├── memory.py            # ConversationMemory — history log + slot store
│   └── bot.py               # CapitalIQBot class — orchestration + main loop
│
├── notebooks/
│   └── project1_chatbot.ipynb   # Complete 19-cell Jupyter Notebook
│
├── tests/
│   ├── __init__.py
│   └── test_chatbot.py      # 35 unit tests across all modules
│
├── assets/
│   └── demo_screenshot.png  # Add your terminal screenshot here
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/decodelabs-ai-project1.git
cd decodelabs-ai-project1
```

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
> The core chatbot uses only the Python Standard Library. Jupyter and pytest are optional extras.

### 4. Run the Chatbot (Interactive Terminal)

```bash
python -m chatbot.bot
```

Or from the project root:
```bash
python chatbot/bot.py
```

### 5. Run the Jupyter Notebook

```bash
jupyter notebook notebooks/project1_chatbot.ipynb
```

### 6. Run the Unit Tests

```bash
# With pytest (recommended):
python -m pytest tests/ -v

# With unittest:
python -m unittest discover tests/ -v
```

---

## 💬 Sample Conversation

```
══════════════════════════════════════════════════════════════
  💹  CapitalIQ  v2.0.0  —  Finance & Investment Advisory
  🎓  DecodeLabs AI Internship  |  Batch 2026
  👤  Built by Ali Ahmad
══════════════════════════════════════════════════════════════

  You: hello

  CapitalIQ: Hello! I'm CapitalIQ, your personal Finance & Investment Advisor.
  I can help you understand stocks, bonds, mutual funds, ETFs, crypto,
  inflation, compound interest, diversification, risk, portfolio strategy,
  budgeting, emergency funds, SIPs, and IPOs.
  What financial topic would you like to explore today?

  You: my name is Ali

  CapitalIQ: Nice to meet you, Ali! I'm CapitalIQ, your Finance & Investment
  Advisor. Hello! I'm CapitalIQ...

  You: what is compound interest?

  CapitalIQ: Compound interest is interest earned on both your original
  principal AND the accumulated interest from previous periods...
  [DEBUG] intent='compound_interest' | confidence=0.250

  You: tell me about ETFs

  CapitalIQ: An ETF (Exchange-Traded Fund) is an investment fund that trades
  on a stock exchange like an individual stock, throughout the trading day...

  You: quit

  CapitalIQ: It was a pleasure advising you today!...

  ══ SESSION TRANSCRIPT — CapitalIQ ══════════════════════════
  Total turns: 4
  ═════════════════════════════════════════════════════════════
```

---

## 🧠 Knowledge Base

CapitalIQ covers **16 financial intents** with over **80 synonym/alias mappings**:

| Intent | Example Triggers |
|--------|-----------------|
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

## ⚙️ Phase Architecture

### Phase 1 — The Logic Engine
- Multi-keyword intent matching using `KEYWORD_INTENT_MAP`
- O(1) synonym resolution via `SYNONYM_MAP`
- Fallback response for unrecognized inputs
- Graceful handling of empty, whitespace, and special-character inputs

### Phase 2 — The Production Upgrade
- **Confidence Scoring**: normalized [0.0, 1.0] score; configurable threshold (default 0.15)
- **Conversation Memory**: full session transcript + named entity slot store
- **Name Extraction**: regex-based detection of "my name is", "I am", "call me"
- **Personalization**: uses remembered name in periodic responses
- **Session Summary**: displays history and stats at session end

---

## 🔬 Design Decisions

### Why Dictionary over If-Elif?
An `if-elif` ladder has O(n) complexity — every new rule added increases the worst-case lookup time. A Python dictionary uses a hash table, giving O(1) average-case lookup regardless of scale. In production chatbots with thousands of intents, this architectural choice is non-negotiable.

### The White-Box Advantage
Rule-based systems are fully transparent. Every response can be traced back to: input → sanitized form → matched keyword → intent key → response value. There are no hallucinations, no unpredictable outputs. This traceability is legally required in regulated sectors (finance, healthcare) and is exactly what frameworks like NVIDIA NeMo Guardrails implement as the control layer above LLMs.

### How This Connects to LLM Systems
In hybrid production architectures, this chatbot's role is the **guardrail layer** — it handles high-confidence, known intents at zero cost and near-zero latency. Only when confidence falls below threshold does the query escalate to an LLM fallback. This pattern dramatically reduces LLM API costs while maintaining safety and auditability.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Dependencies | Standard Library only (`re`, `datetime`, `unittest`) |
| Notebook | Jupyter Notebook |
| Testing | `unittest` + `pytest` |
| Architecture | IPO Model, O(1) Hash Map, Modular OOP |

---

## ✅ DecodeLabs Spec Checklist

| Requirement | Satisfied By |
|-------------|-------------|
| Input Loop | `while True` in `CapitalIQBot.run()` — `bot.py` line 139 |
| Sanitization | `sanitize()` in `sanitizer.py` — `.lower().strip()` + regex |
| Knowledge Base | `BASE_KNOWLEDGE` dict (16 intents) in `knowledge_base.py` |
| Fallback | `FALLBACK_RESPONSE` — returned when `score_intents()` finds no match |
| Exit Strategy | `is_exit_command()` in `matcher.py` — `break` in main loop |

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

*Built with discipline and curiosity as part of the DecodeLabs AI Industrial Training Programme, Batch 2026.*

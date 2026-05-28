"""
chatbot/knowledge_base.py
-------------------------
The central intelligence store for CapitalIQ — a Finance & Investment Advisor chatbot.

Author  : Ali Ahmad
Project : DecodeLabs AI Project 1 — Rule-Based AI Chatbot
Batch   : 2026 | Powered by DecodeLabs

Design Rationale:
    All domain knowledge is stored in pure Python dictionaries (hash maps).
    This achieves O(1) constant-time lookup — the architectural core of this
    project and the reason we reject the O(n) if-elif ladder anti-pattern.

    Three dictionaries serve distinct roles in the pipeline:
        BASE_KNOWLEDGE      — canonical intent → full response string
        SYNONYM_MAP         — any user phrase → canonical intent key
        KEYWORD_INTENT_MAP  — single token → canonical intent key (for scoring)
"""

# ===========================================================================
# PHASE 1: BASE KNOWLEDGE BASE
# ---------------------------------------------------------------------------
# Each key is a canonical intent name. Each value is the bot's full response.
# This is the 'source of truth' — all other maps resolve back to these keys.
# ===========================================================================

BASE_KNOWLEDGE: dict[str, str] = {
    "greeting": (
        "Hello! I'm CapitalIQ, your personal Finance & Investment Advisor. "
        "I can help you understand stocks, bonds, mutual funds, ETFs, crypto, "
        "inflation, compound interest, diversification, risk, portfolio strategy, "
        "budgeting, emergency funds, SIPs, and IPOs. "
        "What financial topic would you like to explore today?"
    ),
    "goodbye": (
        "It was a pleasure advising you today! Remember: the best investment you "
        "can make is in yourself — keep learning, stay disciplined, and let "
        "compound interest work in your favour. Goodbye and invest wisely!"
    ),
    "stocks": (
        "A stock (also called a share or equity) represents fractional ownership "
        "in a company. When you buy stock, you become a shareholder and can profit "
        "in two ways: (1) capital appreciation — the stock price rises above your "
        "purchase price — and (2) dividends — a share of the company's profits paid "
        "periodically. Stocks are traded on exchanges like NYSE, NASDAQ, or BSE. "
        "They carry higher risk than bonds but historically deliver higher long-term "
        "returns. Key metrics to evaluate a stock include P/E ratio, EPS, revenue "
        "growth, and debt-to-equity ratio."
    ),
    "bonds": (
        "A bond is a fixed-income debt instrument. When you buy a bond, you are "
        "lending money to a government or corporation. In return, the issuer promises "
        "to pay you a fixed interest rate (called the coupon) at regular intervals "
        "and return your principal at the maturity date. "
        "Government bonds (like US Treasuries or Indian G-Secs) are the safest; "
        "corporate bonds offer higher yields but carry default risk. "
        "Bonds are essential for portfolio stability and are negatively correlated "
        "with stocks, providing a natural hedge in a diversified portfolio."
    ),
    "mutual_funds": (
        "A mutual fund pools capital from many investors to build a professionally "
        "managed, diversified portfolio of stocks, bonds, or other assets. "
        "Each investor holds 'units' proportional to their contribution. "
        "Key advantages: instant diversification, professional management, and low "
        "minimum investment. Key cost to watch: the Expense Ratio — the annual fee "
        "charged by the fund house, expressed as a percentage of AUM. "
        "Types include equity funds, debt funds, hybrid funds, and index funds. "
        "Compare the fund's CAGR against its benchmark before investing."
    ),
    "etf": (
        "An ETF (Exchange-Traded Fund) is an investment fund that trades on a stock "
        "exchange like an individual stock, throughout the trading day. "
        "Unlike mutual funds (priced once daily at NAV), ETFs have real-time pricing. "
        "Most ETFs passively track an index (e.g., Nifty 50, S&P 500), resulting in "
        "very low expense ratios — often below 0.1%. This makes them one of the most "
        "cost-efficient investment vehicles available. "
        "ETFs are ideal for long-term, low-cost, passive investors who want broad "
        "market exposure without stock-picking risk."
    ),
    "crypto": (
        "Cryptocurrency is a decentralized digital currency secured by cryptography "
        "and recorded on a distributed ledger called a blockchain. "
        "Bitcoin (BTC) is the first and largest by market cap; Ethereum (ETH) "
        "introduced smart contracts, enabling DeFi and NFTs. "
        "Crypto offers potentially high returns but comes with extreme volatility, "
        "regulatory uncertainty, and no intrinsic value guarantee. "
        "Rule of thumb: never allocate more than 5–10% of your portfolio to crypto, "
        "and only invest what you can afford to lose entirely. "
        "Always use regulated exchanges and secure hardware wallets."
    ),
    "inflation": (
        "Inflation is the rate at which the general price level of goods and services "
        "rises over time, eroding the purchasing power of money. "
        "Central banks (the US Fed, RBI, ECB) target around 2% annual inflation. "
        "High inflation means ₹100 today will buy less next year. "
        "As an investor, you must target returns that outpace inflation — this is why "
        "simply holding cash in a savings account is rarely sufficient. "
        "Asset classes that historically beat inflation include equities, real estate, "
        "gold, and inflation-linked bonds (TIPS in the US, IIBs in India)."
    ),
    "compound_interest": (
        "Compound interest is interest earned on both your original principal AND the "
        "accumulated interest from previous periods. This creates exponential — not "
        "linear — growth. The formula is: A = P × (1 + r/n)^(n×t), where P = principal, "
        "r = annual rate, n = compounding frequency, t = time in years. "
        "Example: ₹1,00,000 at 12% p.a. compounded monthly for 20 years → ₹9,89,255. "
        "The single most powerful insight: starting early dwarfs the impact of investing "
        "more later. A 25-year-old investing ₹5,000/month will vastly outperform a "
        "35-year-old investing ₹10,000/month by retirement — thanks to compounding."
    ),
    "diversification": (
        "Diversification is the risk-management strategy of spreading investments "
        "across multiple uncorrelated asset classes, sectors, and geographies. "
        "The core principle: when one asset falls, others may rise or hold steady, "
        "smoothing overall portfolio volatility. "
        "A well-diversified portfolio typically includes: domestic equities, "
        "international equities, government bonds, corporate bonds, real estate (REITs), "
        "gold/commodities, and cash equivalents. "
        "Correlation matters: don't just diversify across stocks — diversify across "
        "asset classes with low or negative correlation to each other."
    ),
    "risk": (
        "Investment risk is the probability that your actual return differs from your "
        "expected return — including the risk of permanent capital loss. "
        "Key types: (1) Market Risk — broad market declines affect all holdings. "
        "(2) Credit/Default Risk — the issuer fails to repay. "
        "(3) Liquidity Risk — you cannot sell when needed. "
        "(4) Inflation Risk — returns fail to outpace inflation. "
        "(5) Concentration Risk — too much in one asset or sector. "
        "Your risk tolerance is shaped by your time horizon, income stability, and "
        "emotional capacity to withstand drawdowns. Use it to determine your asset "
        "allocation between high-risk equities and lower-risk bonds/cash."
    ),
    "portfolio": (
        "An investment portfolio is the complete collection of all your financial assets: "
        "stocks, bonds, ETFs, mutual funds, real estate, gold, crypto, and cash. "
        "Portfolio construction starts with your Investment Policy Statement (IPS): "
        "goals, time horizon, return expectations, and risk tolerance. "
        "A classic starting allocation is the 60/40 portfolio — 60% equities, 40% bonds. "
        "Aggressive investors (longer horizon) may go 80/20 or 100% equities. "
        "Critical habit: rebalance annually. If equities surge and now represent 75% of "
        "your portfolio, sell some and buy bonds to restore your target allocation."
    ),
    "budgeting": (
        "Budgeting is the foundational discipline of personal finance. Without it, "
        "investing is impossible at any meaningful scale. "
        "The most popular framework is the 50/30/20 Rule: allocate 50% of net income "
        "to Needs (rent, food, utilities), 30% to Wants (dining, entertainment, travel), "
        "and 20% to Savings & Investments. "
        "Before investing, prioritize in this order: "
        "(1) Build a 3–6 month emergency fund. "
        "(2) Pay off all high-interest debt (credit cards, personal loans). "
        "(3) Then invest aggressively and consistently. "
        "Track every expense for at least 3 months to understand your true spending patterns."
    ),
    "emergency_fund": (
        "An emergency fund is 3 to 6 months of your total living expenses held in a "
        "highly liquid, principal-safe account — such as a high-yield savings account "
        "or a liquid mutual fund. "
        "Its purpose: absorb financial shocks (job loss, medical emergency, major repair) "
        "without forcing you to liquidate investments at a bad time. "
        "This is Step Zero of investing — it must be established BEFORE you put money "
        "into stocks, mutual funds, or any illiquid asset. "
        "Size your fund based on job stability, number of dependants, and monthly expenses. "
        "Keep it in a separate account to avoid spending it impulsively."
    ),
    "sip": (
        "A SIP (Systematic Investment Plan) is a disciplined investment method where "
        "you invest a fixed amount into a mutual fund at regular intervals (typically monthly). "
        "SIPs leverage Rupee-Cost Averaging (RCA): you buy more units when prices are low "
        "and fewer when prices are high, reducing your average cost per unit over time. "
        "This removes the need to 'time the market' — arguably the most dangerous habit "
        "for retail investors. "
        "Even ₹500/month invested via SIP over 20 years at 12% CAGR grows to ~₹4.99 lakhs. "
        "Automate your SIP on salary day so investing happens before discretionary spending."
    ),
    "ipo": (
        "An IPO (Initial Public Offering) is the process by which a private company offers "
        "its shares to the public on a stock exchange for the first time. "
        "IPOs can generate buzz and short-term gains (especially on listing day), but they "
        "carry significant risk: limited financial history, lock-up period expiry selling "
        "pressure, and often inflated valuations driven by hype. "
        "Before applying for an IPO: read the DRHP (Draft Red Herring Prospectus) — it "
        "contains the company's financials, risk factors, and use of proceeds. "
        "Ask: is the valuation reasonable compared to listed peers? Is the business model "
        "proven? Is management credible? Never invest in an IPO based on hype alone."
    ),
}


# ===========================================================================
# PHASE 2: SYNONYM / ALIAS MAP
# ---------------------------------------------------------------------------
# Maps EVERY possible user phrase to a canonical intent key.
# This is the 'synonym resolution layer' — it runs BEFORE keyword scoring.
# Exact phrase match here achieves O(1) lookup for common phrasings.
# ===========================================================================

SYNONYM_MAP: dict[str, str] = {
    # ── Greeting ──────────────────────────────────────────────────────────
    "hi": "greeting",
    "hello": "greeting",
    "hey": "greeting",
    "sup": "greeting",
    "good morning": "greeting",
    "good afternoon": "greeting",
    "good evening": "greeting",
    "howdy": "greeting",
    "yo": "greeting",
    "start": "greeting",
    "help": "greeting",
    "what can you do": "greeting",

    # ── Goodbye ───────────────────────────────────────────────────────────
    "bye": "goodbye",
    "goodbye": "goodbye",
    "quit": "goodbye",
    "exit": "goodbye",
    "see you": "goodbye",
    "later": "goodbye",
    "take care": "goodbye",
    "done": "goodbye",
    "close": "goodbye",
    "tata": "goodbye",
    "ciao": "goodbye",

    # ── Stocks ────────────────────────────────────────────────────────────
    "stocks": "stocks",
    "stock": "stocks",
    "share": "stocks",
    "shares": "stocks",
    "equity": "stocks",
    "equities": "stocks",
    "stock market": "stocks",
    "how to buy stocks": "stocks",
    "what are stocks": "stocks",
    "what is a stock": "stocks",
    "invest in stocks": "stocks",
    "pe ratio": "stocks",

    # ── Bonds ─────────────────────────────────────────────────────────────
    "bond": "bonds",
    "bonds": "bonds",
    "fixed income": "bonds",
    "government bond": "bonds",
    "government bonds": "bonds",
    "treasury": "bonds",
    "coupon": "bonds",
    "debenture": "bonds",
    "debentures": "bonds",
    "gsec": "bonds",
    "g-sec": "bonds",
    "what are bonds": "bonds",

    # ── Mutual Funds ──────────────────────────────────────────────────────
    "mutual fund": "mutual_funds",
    "mutual funds": "mutual_funds",
    "fund": "mutual_funds",
    "managed fund": "mutual_funds",
    "expense ratio": "mutual_funds",
    "nav": "mutual_funds",
    "what is a mutual fund": "mutual_funds",
    "how to invest in mutual funds": "mutual_funds",

    # ── ETF ───────────────────────────────────────────────────────────────
    "etf": "etf",
    "etfs": "etf",
    "exchange traded fund": "etf",
    "index fund": "etf",
    "index etf": "etf",
    "nifty etf": "etf",
    "nifty 50 etf": "etf",
    "s&p etf": "etf",
    "what is an etf": "etf",
    "passive investing": "etf",

    # ── Crypto ────────────────────────────────────────────────────────────
    "crypto": "crypto",
    "cryptocurrency": "crypto",
    "bitcoin": "crypto",
    "btc": "crypto",
    "ethereum": "crypto",
    "eth": "crypto",
    "blockchain": "crypto",
    "defi": "crypto",
    "nft": "crypto",
    "altcoin": "crypto",
    "altcoins": "crypto",
    "digital currency": "crypto",
    "web3": "crypto",
    "solana": "crypto",

    # ── Inflation ─────────────────────────────────────────────────────────
    "inflation": "inflation",
    "purchasing power": "inflation",
    "price rise": "inflation",
    "cost of living": "inflation",
    "deflation": "inflation",
    "cpi": "inflation",
    "what is inflation": "inflation",
    "how to beat inflation": "inflation",

    # ── Compound Interest ─────────────────────────────────────────────────
    "compound interest": "compound_interest",
    "compounding": "compound_interest",
    "compound": "compound_interest",
    "interest": "compound_interest",
    "time value of money": "compound_interest",
    "power of compounding": "compound_interest",
    "eighth wonder": "compound_interest",
    "how does compounding work": "compound_interest",

    # ── Diversification ───────────────────────────────────────────────────
    "diversification": "diversification",
    "diversify": "diversification",
    "asset allocation": "diversification",
    "asset classes": "diversification",
    "spread investments": "diversification",

    # ── Risk ──────────────────────────────────────────────────────────────
    "risk": "risk",
    "investment risk": "risk",
    "market risk": "risk",
    "volatility": "risk",
    "risk tolerance": "risk",
    "risk management": "risk",
    "safe investment": "risk",
    "how risky is investing": "risk",

    # ── Portfolio ─────────────────────────────────────────────────────────
    "portfolio": "portfolio",
    "investment portfolio": "portfolio",
    "my investments": "portfolio",
    "holdings": "portfolio",
    "rebalancing": "portfolio",
    "asset mix": "portfolio",
    "60 40 portfolio": "portfolio",
    "how to build a portfolio": "portfolio",

    # ── Budgeting ─────────────────────────────────────────────────────────
    "budget": "budgeting",
    "budgeting": "budgeting",
    "how to budget": "budgeting",
    "50 30 20": "budgeting",
    "50/30/20": "budgeting",
    "manage money": "budgeting",
    "money management": "budgeting",
    "personal finance": "budgeting",
    "financial planning": "budgeting",
    "track expenses": "budgeting",

    # ── Emergency Fund ────────────────────────────────────────────────────
    "emergency fund": "emergency_fund",
    "emergency savings": "emergency_fund",
    "rainy day fund": "emergency_fund",
    "liquid savings": "emergency_fund",
    "savings": "emergency_fund",
    "how much to save": "emergency_fund",

    # ── SIP ───────────────────────────────────────────────────────────────
    "sip": "sip",
    "systematic investment plan": "sip",
    "monthly investment": "sip",
    "rupee cost averaging": "sip",
    "recurring investment": "sip",
    "how to start sip": "sip",

    # ── IPO ───────────────────────────────────────────────────────────────
    "ipo": "ipo",
    "initial public offering": "ipo",
    "new stock listing": "ipo",
    "public listing": "ipo",
    "drhp": "ipo",
    "going public": "ipo",
    "should i invest in ipo": "ipo",
}


# ===========================================================================
# KEYWORD → INTENT MAP (for multi-keyword token scoring)
# ---------------------------------------------------------------------------
# Each individual token maps to the intent it signals.
# The matcher counts votes per intent across all tokens in the user's message.
# ===========================================================================

KEYWORD_INTENT_MAP: dict[str, str] = {
    # Greeting
    "hi": "greeting",       "hello": "greeting",    "hey": "greeting",
    "help": "greeting",     "start": "greeting",    "howdy": "greeting",

    # Goodbye
    "bye": "goodbye",       "exit": "goodbye",      "quit": "goodbye",
    "goodbye": "goodbye",   "done": "goodbye",      "close": "goodbye",

    # Stocks
    "stock": "stocks",      "stocks": "stocks",     "share": "stocks",
    "shares": "stocks",     "equity": "stocks",     "equities": "stocks",
    "nasdaq": "stocks",     "nyse": "stocks",       "bse": "stocks",
    "dividend": "stocks",   "shareholder": "stocks",

    # Bonds
    "bond": "bonds",        "bonds": "bonds",       "treasury": "bonds",
    "coupon": "bonds",      "debenture": "bonds",   "gsec": "bonds",
    "maturity": "bonds",    "yield": "bonds",       "fixed": "bonds",

    # Mutual Funds
    "mutual": "mutual_funds",   "fund": "mutual_funds",
    "managed": "mutual_funds",  "expense": "mutual_funds",
    "nav": "mutual_funds",      "aum": "mutual_funds",

    # ETF
    "etf": "etf",           "etfs": "etf",          "index": "etf",
    "traded": "etf",        "passive": "etf",       "nifty": "etf",

    # Crypto
    "crypto": "crypto",     "bitcoin": "crypto",    "btc": "crypto",
    "ethereum": "crypto",   "eth": "crypto",        "blockchain": "crypto",
    "nft": "crypto",        "defi": "crypto",       "altcoin": "crypto",
    "cryptocurrency": "crypto",                     "web3": "crypto",

    # Inflation
    "inflation": "inflation",   "deflation": "inflation",
    "purchasing": "inflation",  "cpi": "inflation",
    "prices": "inflation",      "eroding": "inflation",

    # Compound Interest
    "compound": "compound_interest",    "compounding": "compound_interest",
    "interest": "compound_interest",    "principal": "compound_interest",
    "exponential": "compound_interest",

    # Diversification
    "diversification": "diversification",   "diversify": "diversification",
    "allocation": "diversification",        "spread": "diversification",
    "uncorrelated": "diversification",      "hedge": "diversification",

    # Risk
    "risk": "risk",         "volatility": "risk",   "risky": "risk",
    "safe": "risk",         "drawdown": "risk",     "default": "risk",
    "liquidity": "risk",    "tolerance": "risk",

    # Portfolio
    "portfolio": "portfolio",   "holdings": "portfolio",
    "rebalancing": "portfolio", "rebalance": "portfolio",
    "allocation": "portfolio",  "60/40": "portfolio",

    # Budgeting
    "budget": "budgeting",      "budgeting": "budgeting",
    "manage": "budgeting",      "expenses": "budgeting",
    "spending": "budgeting",    "income": "budgeting",
    "finance": "budgeting",     "planning": "budgeting",

    # Emergency Fund
    "emergency": "emergency_fund",  "savings": "emergency_fund",
    "rainy": "emergency_fund",      "liquid": "emergency_fund",
    "cushion": "emergency_fund",    "safety": "emergency_fund",

    # SIP
    "sip": "sip",           "systematic": "sip",    "monthly": "sip",
    "recurring": "sip",     "rupee": "sip",         "averaging": "sip",

    # IPO
    "ipo": "ipo",           "initial": "ipo",       "offering": "ipo",
    "listing": "ipo",       "drhp": "ipo",          "prospectus": "ipo",
    "public": "ipo",
}


# ===========================================================================
# FALLBACK & EXIT CONFIGURATION
# ===========================================================================

FALLBACK_RESPONSE: str = (
    "I'm not quite sure I understood that. I specialise in: stocks, bonds, "
    "mutual funds, ETFs, crypto, inflation, compound interest, diversification, "
    "risk, portfolio strategy, budgeting, emergency funds, SIPs, and IPOs. "
    "Could you rephrase your question or ask about one of those topics?"
)

EXIT_KEYWORDS: set[str] = {"exit", "quit", "bye", "goodbye", "done", "close", "tata", "ciao"}

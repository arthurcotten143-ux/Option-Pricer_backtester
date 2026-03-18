"""
Options Pricer — Streamlit
Run with: streamlit run Option-Pricer_Backtester.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Options Pricer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "app"

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #000000 !important; }
    .block-container { padding-top: 1rem !important; background-color: #000000 !important; }
    [data-testid="stSidebar"] { background-color: #0a0a0a !important; }
    [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] { background-color: #000000 !important; }
    h1 { color: #4a9eff !important; font-family: monospace !important; font-weight: bold !important;
         font-size: 1.73rem !important; text-shadow: 0 0 8px rgba(74,158,255,0.4); }
    h2 { color: #4a9eff !important; font-family: monospace !important; font-weight: normal !important; font-size: 1.3rem !important; }
    h3 { color: #06b6d4 !important; font-family: monospace !important; font-weight: normal !important; font-size: 1.05rem !important; }
    h4 { color: #8b5cf6 !important; font-family: monospace !important; font-weight: normal !important; font-size: 0.95rem !important; }
    p, span, div, label, .stMarkdown { color: #ffffff !important; font-weight: normal !important; font-size: 0.72rem !important; }
    input[type="number"], input[type="text"], .stNumberInput input, .stTextInput input {
        background-color: #0a0a0a !important; color: #ffffff !important; border: 2px solid #4a9eff !important; }
    .stSelectbox > div > div, select {
        background-color: #0a0a0a !important; color: #ffffff !important; border: 2px solid #4a9eff !important; }
    .stSelectbox div[data-baseweb="select"] > div, .stSelectbox ul, .stSelectbox li,
    [role="listbox"], [role="option"] { background-color: #0a0a0a !important; color: #ffffff !important; }
    .stSelectbox li:hover, [role="option"]:hover { background-color: #1a1a1a !important; color: #4a9eff !important; }
    div[data-testid="metric-container"] {
        background-color: #000000 !important; border: 2px solid #4a9eff !important;
        border-radius: 10px !important; padding: 13px !important; box-shadow: 0 0 15px rgba(74,158,255,0.3) !important; }
    div[data-testid="metric-container"] label, div[data-testid="metric-container"] label p {
        color: #4a9eff !important; font-weight: normal !important; font-size: 0.72rem !important; }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div, div[data-testid="stMetricValue"] p {
        color: #ffffff !important; font-weight: normal !important; font-size: 1.3rem !important; }
    div[data-testid="stMetricDelta"] { color: #4a9eff !important; }
    .stAlert, .stSuccess, .stWarning { background-color: #0a0a0a !important; border: 2px solid #4a9eff !important; }
    .stAlert p, .stSuccess p, .stWarning p { color: #ffffff !important; }
    .dataframe { font-size: 0.72rem !important; font-family: monospace !important;
                 background-color: #000000 !important; border: 2px solid #4a9eff !important; }
    .dataframe th { background-color: #0a0a0a !important; color: #4a9eff !important;
                    border: 1px solid #4a9eff !important; padding: 6px !important; }
    .dataframe td { color: #ffffff !important; background-color: #000000 !important;
                    border: 1px solid #333333 !important; padding: 6px !important; }
    .author-link { color: #888888 !important; font-size: 0.72rem; font-family: monospace;
                   margin-top: -10px; margin-bottom: 15px; }
    .author-link a { color: #4a9eff !important; text-decoration: none; }
    .author-link a:hover { color: #60a5fa !important; text-decoration: underline; }
    [data-testid="stSidebar"] .stButton > button {
        background-color: #0a0a0a !important; color: #4a9eff !important;
        border: 1px solid #4a9eff !important; font-family: monospace !important; font-size: 0.72rem !important; }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #0c1a2e !important; border-color: #60a5fa !important; color: #60a5fa !important; }
    .formula-box {
        background-color: #0a0a0a; border: 1px solid #2a4a6b; border-left: 3px solid #4a9eff;
        border-radius: 6px; padding: 14px 18px; margin: 10px 0; font-family: monospace;
        font-size: 0.85rem; color: #e5e7eb; line-height: 1.8;
    }
    .tag-green  { background:#052e16; color:#10b981; border:1px solid #10b981; border-radius:4px; padding:2px 7px; font-size:0.68rem; font-family:monospace; }
    .tag-blue   { background:#0c1a2e; color:#4a9eff; border:1px solid #4a9eff; border-radius:4px; padding:2px 7px; font-size:0.68rem; font-family:monospace; }
    .tag-purple { background:#1a0a2e; color:#8b5cf6; border:1px solid #8b5cf6; border-radius:4px; padding:2px 7px; font-size:0.68rem; font-family:monospace; }
    .tag-yellow { background:#1c1200; color:#f59e0b; border:1px solid #f59e0b; border-radius:4px; padding:2px 7px; font-size:0.68rem; font-family:monospace; }
    .section-divider { border: none; border-top: 1px solid #1e2a38; margin: 28px 0; }
    /* Markdown tables */
    table { border-collapse: collapse !important; width: 100% !important; background-color: #000000 !important; }
    thead tr { background-color: #0a0a0a !important; }
    thead th { color: #4a9eff !important; font-family: monospace !important; font-weight: normal !important;
               border-bottom: 1px solid #2a4a6b !important; padding: 6px 10px !important;
               font-size: 0.72rem !important; background-color: #0a0a0a !important; }
    tbody tr { background-color: #000000 !important; }
    tbody tr:hover { background-color: #080808 !important; }
    tbody td { color: #e5e7eb !important; font-family: monospace !important; font-size: 0.72rem !important;
               border-bottom: 1px solid #1a1a1a !important; padding: 5px 10px !important;
               background-color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB STYLE ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#000000", "axes.facecolor": "#0a0a0a",
    "axes.edgecolor": "#4a9eff",   "axes.labelcolor": "#4a9eff",
    "text.color": "#e5e7eb",       "xtick.color": "#e5e7eb",
    "ytick.color": "#e5e7eb",      "grid.color": "#1e2a38",
    "grid.linewidth": 0.35,        "grid.alpha": 0.5,
    "font.family": "monospace",    "font.weight": "normal",
    "xtick.major.width": 0.5,      "ytick.major.width": 0.5,
})

BG, PANEL = "#000000", "#0a0a0a"
ACCENT, GREEN, RED = "#4a9eff", "#10b981", "#ef4444"
YELLOW, PURPLE, CYAN = "#f59e0b", "#8b5cf6", "#06b6d4"
GRAY, TEXT, TITLE = "#6b7280", "#e5e7eb", "#4a9eff"

# ─── CORE FUNCTIONS ───────────────────────────────────────────────────────────

def bs(S, K, T, r, sigma, q=0.0, opt="call"):
    if T <= 1e-10:
        return max(S-K, 0) if opt=="call" else max(K-S, 0)
    if sigma <= 1e-10:
        return max(S*np.exp(-q*T)-K*np.exp(-r*T), 0) if opt=="call" \
               else max(K*np.exp(-r*T)-S*np.exp(-q*T), 0)
    d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt == "call":
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def greeks(S, K, T, r, sigma, q=0.0, opt="call"):
    if T <= 1e-10 or sigma <= 1e-10:
        return {k: 0.0 for k in ["delta","gamma","vega","theta","rho"]}
    d1 = (np.log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T); nd1 = norm.pdf(d1)
    if opt == "call":
        delta = np.exp(-q*T)*norm.cdf(d1)
        theta = (-(S*np.exp(-q*T)*nd1*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2) + q*S*np.exp(-q*T)*norm.cdf(d1)) / 365
        rho   = K*T*np.exp(-r*T)*norm.cdf(d2) / 100
    else:
        delta = -np.exp(-q*T)*norm.cdf(-d1)
        theta = (-(S*np.exp(-q*T)*nd1*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2) - q*S*np.exp(-q*T)*norm.cdf(-d1)) / 365
        rho   = -K*T*np.exp(-r*T)*norm.cdf(-d2) / 100
    gamma = np.exp(-q*T)*nd1 / (S*sigma*np.sqrt(T))
    vega  = S*np.exp(-q*T)*nd1*np.sqrt(T) / 100
    return {"delta":delta,"gamma":gamma,"vega":vega,"theta":theta,"rho":rho}

def prob_itm(S, K, T, r, sigma, q=0.0, opt="call"):
    if T <= 1e-10 or sigma <= 1e-10:
        return 1.0 if (opt=="call" and S>K) or (opt=="put" and S<K) else 0.0
    d2 = (np.log(S/K)+(r-q-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.cdf(d2) if opt=="call" else norm.cdf(-d2)

def implied_volatility(market_price, S, K, T, r, q=0.0, opt="call"):
    if T <= 1e-10: return np.nan
    if market_price < (max(S-K,0) if opt=="call" else max(K-S,0)) * 0.99: return np.nan
    try:
        return brentq(lambda sig: bs(S,K,T,r,sig,q,opt)-market_price, 0.001, 5.0, maxiter=100)
    except:
        return np.nan

@st.cache_data(ttl=300)
def monte_carlo_pricer_cached(S, K, T, r, sigma, q, opt, n_sims, n_steps, antithetic, seed):
    np.random.seed(seed); dt = T/n_steps
    all_payoffs, sample_paths = [], None
    for batch in range(int(np.ceil(n_sims/100000))):
        nb = min(100000, n_sims-batch*100000)
        n  = nb//2 if antithetic else nb
        Z  = np.random.standard_normal((n, n_steps))
        if antithetic: Z = np.concatenate([Z,-Z], axis=0)
        S_T = S * np.exp(np.sum((r-q-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z, axis=1))
        if batch==0: sample_paths = S_T[:min(1000,n_sims)].copy()
        all_payoffs.append(np.maximum(S_T-K,0) if opt=="call" else np.maximum(K-S_T,0))
        del Z, S_T
    all_payoffs = np.concatenate(all_payoffs)
    price = np.exp(-r*T)*np.mean(all_payoffs)
    se    = np.exp(-r*T)*np.std(all_payoffs)/np.sqrt(len(all_payoffs))
    g     = greeks(S,K,T,r,sigma,q,opt)
    return {"price":price,"std_error":se,"paths":sample_paths,**g}

@st.cache_data(ttl=300)
def backtest_strategy_cached(strategy, S0, K, T, r, sigma, q, n_days, n_sims):
    np.random.seed(42); dt = T/n_days
    Z   = np.random.standard_normal((n_sims, n_days))
    S_f = S0*np.exp(np.sum((r-q-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z, axis=1))
    rows = []
    for Se in S_f:
        if   strategy=="long_call":      pnl = max(Se-K,0) - bs(S0,K,T,r,sigma,q,"call")
        elif strategy=="long_put":       pnl = max(K-Se,0) - bs(S0,K,T,r,sigma,q,"put")
        elif strategy=="covered_call":   pnl = (Se-S0)+bs(S0,K,T,r,sigma,q,"call")-max(Se-K,0)
        elif strategy=="protective_put": pnl = (Se-S0)-bs(S0,K,T,r,sigma,q,"put")+max(K-Se,0)
        elif strategy=="straddle":       pnl = max(Se-K,0)+max(K-Se,0)-bs(S0,K,T,r,sigma,q,"call")-bs(S0,K,T,r,sigma,q,"put")
        elif strategy=="strangle":
            Kc,Kp = K*1.05,K*0.95
            pnl   = max(Se-Kc,0)+max(Kp-Se,0)-bs(S0,Kc,T,r,sigma,q,"call")-bs(S0,Kp,T,r,sigma,q,"put")
        rows.append({"final_spot":Se,"pnl":pnl,"return_pct":(pnl/S0)*100})
    return pd.DataFrame(rows)

# ─── DELTA HEDGING ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def delta_hedge_simulation(S0, K, T, r, sigma, q, opt, n_days, n_paths, freq):
    """
    Simulate dynamic delta hedging of a long option position.
    At each rebalancing step: adjust stock holding to -Delta (hedge).
    Final P&L = option payoff + stock P&L + cash account (risk-free).
    Residual P&L ≈ Gamma P&L (the cost/benefit of hedging imperfectly).
    """
    np.random.seed(42)
    freq_map = {"Daily": 1, "Weekly": 5, "At expiry": n_days}
    rebal_every = freq_map.get(freq, 1)

    dt = T / n_days
    hedge_pnls, unhedged_pnls, delta_paths, gamma_pnls = [], [], [], []

    for _ in range(n_paths):
        # Simulate full price path
        Z    = np.random.standard_normal(n_days)
        logS = np.log(S0) + np.cumsum((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
        path = np.concatenate([[S0], np.exp(logS)])  # length n_days+1

        # Option entry cost
        entry_cost = bs(S0, K, T, r, sigma, q, opt)

        # Init hedge
        cash       = -entry_cost   # paid for option
        stock_pos  = 0.0           # shares held (will be set to -Delta)
        deltas     = []

        for i in range(n_days):
            S_now    = path[i]
            T_rem    = T - i * dt
            if T_rem < 1e-8:
                break
            delta_now = greeks(S_now, K, T_rem, r, sigma, q, opt)["delta"]
            deltas.append(delta_now)

            # Rebalance only at specified frequency
            if i % rebal_every == 0:
                new_stock = -delta_now           # hedge: short Delta shares for long call
                trade     = new_stock - stock_pos
                cash     -= trade * S_now        # buy/sell stock at current price
                stock_pos = new_stock

            # Accrue risk-free interest on cash daily
            cash *= np.exp(r * dt)

        # At expiry
        S_T      = path[-1]
        payoff   = max(S_T - K, 0) if opt == "call" else max(K - S_T, 0)
        cash    += stock_pos * S_T   # unwind stock position
        cash    += payoff             # receive option payoff

        hedge_pnl    = cash           # total hedged P&L
        unhedged_pnl = payoff - entry_cost

        hedge_pnls.append(hedge_pnl)
        unhedged_pnls.append(unhedged_pnl)
        delta_paths.append(deltas)
        gamma_pnls.append(hedge_pnl - unhedged_pnl * 0)   # residual = hedge P&L

    return {
        "hedge_pnls":    np.array(hedge_pnls),
        "unhedged_pnls": np.array(unhedged_pnls),
        "delta_paths":   delta_paths,
        "entry_cost":    entry_cost,
    }

# ─── PLOT HELPERS ─────────────────────────────────────────────────────────────

def sty(ax, title, xl, yl):
    ax.set_title(title, color=TITLE, fontsize=8.5, pad=7, fontweight="normal", loc="left")
    ax.set_xlabel(xl, color="#8eafc2", fontsize=7.5, labelpad=6)
    ax.set_ylabel(yl, color="#8eafc2", fontsize=7.5, labelpad=6)
    ax.grid(True, alpha=0.18, linewidth=0.3, linestyle="--")
    ax.tick_params(labelsize=7, colors="#9ca3af", width=0.5, length=3, pad=4)
    for sp in ax.spines.values(): sp.set_linewidth(0.6); sp.set_edgecolor("#2a4a6b")
    ax.set_axisbelow(True)

def annotate_be(ax, val, ymin, color=GREEN):
    ax.annotate(f"BE ${val:.2f}", xy=(val, ymin), fontsize=4.5, color=color,
                fontfamily="monospace", ha="center", va="top", alpha=0.85)

def vline(ax, x, label, color, ymin, ymax):
    ax.axvline(x, color=color, lw=0.5, linestyle="--", alpha=0.6)
    ax.annotate(label, xy=(x, ymin+(ymax-ymin)*0.04), fontsize=4.5, color=color,
                fontfamily="monospace", ha="center", va="bottom", rotation=90, alpha=0.85)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ◈ OPTIONS PRICER")
    st.markdown("---")

    # ── DOCS BUTTON ───────────────────────────────────────────────────────────
    if st.session_state.page == "app":
        if st.button("📐  Formula Reference", use_container_width=True):
            st.session_state.page = "docs"
            st.rerun()
    else:
        if st.button("◀  Back to App", use_container_width=True):
            st.session_state.page = "app"
            st.rerun()

    st.markdown("---")

    if st.session_state.page == "app":
        st.markdown("### Mode")
        mode = st.selectbox("Select mode", ["Pricing", "Implied Volatility", "Backtesting"])

        pricing_method = "Black-Scholes"
        market_price   = 5.0
        strategy       = "long_call"
        backtest_days  = 30
        n_simulations  = 1000
        prem           = 0.0
        n_sims         = 100000
        n_steps        = 252
        antithetic     = True
        seed           = 42

        if mode == "Pricing":
            st.markdown("---")
            st.markdown("### Pricing model")
            pricing_method = st.selectbox("Model", ["Black-Scholes", "Monte Carlo"])

        st.markdown("---")
        st.markdown("### Parameters")
        S     = st.number_input("Spot S ($)",           value=100.0, step=1.0)
        K     = st.number_input("Strike K ($)",         value=100.0, step=1.0)
        T_day = st.number_input("Maturity (days)",      value=30,    step=1,   min_value=1)
        r     = st.number_input("Risk-free rate r (%)", value=5.0,   step=0.1) / 100
        sigma = st.number_input("Volatility σ (%)",     value=20.0,  step=0.5) / 100
        q     = st.number_input("Dividend yield q (%)", value=0.0,   step=0.1) / 100
        opt   = st.radio("Option type", ["call", "put"], horizontal=True)

        if mode == "Pricing":
            prem = st.number_input("Premium paid ($) [opt.]", value=0.0, step=0.01)
            if pricing_method == "Monte Carlo":
                st.markdown("---")
                st.markdown("### Monte Carlo settings")
                n_sims     = st.selectbox("Simulations", [10000,50000,100000,250000], index=2)
                n_steps    = st.selectbox("Time steps",  [50,100,252], index=2)
                antithetic = st.checkbox("Antithetic variates", value=True)
                seed       = st.number_input("Seed", value=42, step=1)

        elif mode == "Implied Volatility":
            st.markdown("---")
            st.markdown("### Market price")
            market_price = st.number_input("Observed price ($)", value=5.0, step=0.01, min_value=0.01)

        elif mode == "Backtesting":
            st.markdown("---")
            st.markdown("### Backtest settings")
            strategy = st.selectbox("Strategy",
                ["long_call","long_put","covered_call","protective_put","straddle","strangle"],
                format_func=lambda x: x.replace('_',' ').title())
            backtest_days = st.slider("Horizon (days)", 1, min(365,T_day), min(T_day,30))
            n_simulations = st.selectbox("Simulations", [100,500,1000,2000], index=2)
            hedge_freq    = st.selectbox("Delta hedge rebalancing", ["Daily","Weekly","At expiry"], index=0)
            hedge_n_paths = st.selectbox("Delta hedge paths", [50,100,250], index=1)

        st.markdown("---")
        run = st.button("⚡  RUN", use_container_width=True, type="primary")

    else:
        S=K=100.0; T_day=30; r=0.05; sigma=0.20; q=0.0; opt="call"
        mode="Pricing"; pricing_method="Black-Scholes"
        prem=n_sims=n_steps=seed=0; antithetic=True
        market_price=5.0; strategy="long_call"; backtest_days=30; n_simulations=1000; run=False
        hedge_freq="Daily"; hedge_n_paths=100

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DOCS
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.page == "docs":

    st.markdown("# 📐 Formula Reference")
    st.markdown('<div class="author-link">by <a href="https://www.linkedin.com/in/arthurcotten/">Arthur Cotten</a> • <a href="https://github.com/arthurcotten">@arthurcotten</a></div>', unsafe_allow_html=True)
    st.markdown("---")

    doc_tab1, doc_tab2, doc_tab3, doc_tab4, doc_tab5 = st.tabs([
        "① Black-Scholes", "② Monte Carlo", "③ Implied Volatility", "④ Backtesting", "⑤ Delta Hedging"
    ])

    # ── BLACK-SCHOLES ─────────────────────────────────────────────────────────
    with doc_tab1:
        st.markdown("## Black-Scholes Model")
        st.markdown("""
The Black-Scholes model (1973) provides a closed-form analytical solution to price European-style options.
It assumes the underlying follows a **Geometric Brownian Motion (GBM)** with constant volatility and no arbitrage.
""")
        st.markdown("### Core Formula")
        st.markdown('<div class="formula-box">'
            '<b>Call price:</b>  C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)<br><br>'
            '<b>Put price:</b>   P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)<br><br>'
            '<b>d1</b> = [ ln(S/K) + (r - q + σ²/2)·T ] / (σ·√T)<br>'
            '<b>d2</b> = d1 - σ·√T<br><br>'
            'N(·) = cumulative standard normal distribution function'
            '</div>', unsafe_allow_html=True)

        st.markdown("### Input Parameters")
        st.markdown("""
| Parameter | Name | Unit | Role |
|---|---|---|---|
| S | Spot price | $ | Current market price of the underlying asset |
| K | Strike price | $ | Price at which the option can be exercised |
| T | Time to maturity | years (= days/365) | Time remaining until expiration |
| r | Risk-free rate | decimal (e.g. 0.05) | Continuously compounded risk-free interest rate |
| σ | Volatility | decimal (e.g. 0.20) | Annualised standard deviation of log-returns — key driver of option price |
| q | Dividend yield | decimal (e.g. 0.02) | Continuous dividend yield paid by the underlying |
""")

        st.markdown("### Output Metrics")
        st.markdown("""
| Metric | Formula / Definition | Interpretation |
|---|---|---|
| Price | C or P from above formula | Theoretical fair value of the option under BS assumptions |
| Break-even | Call: K + premium paid / Put: K - premium paid | Spot level at which you neither gain nor lose at expiry |
| Prob ITM | Call: N(d2) / Put: N(-d2) | Risk-neutral probability the option expires in-the-money |
| Intrinsic Value | Call: max(S-K, 0) / Put: max(K-S, 0) | Value if exercised immediately — minimum floor of option price |
| Time Value | Price - Intrinsic Value | Premium paid for optionality over time — decays as T→0 |
| Moneyness | ITM if S>K (call) / ATM if S≈K / OTM otherwise | Position of the strike relative to the current spot |
""")

        st.markdown("### Greeks")
        st.markdown("""
| Greek | Formula | Range | Interpretation |
|---|---|---|---|
| Delta (Δ) | Call: e^(-qT)·N(d1) / Put: -e^(-qT)·N(-d1) | [-1, 1] | Sensitivity of option price to a $1.00 move in the underlying. An ATM option (Delta ~0.5) gains $0.50 when S rises by $1. |
| Gamma (Γ) | e^(-qT)·N'(d1) / (S·σ·√T) | > 0 | Rate of change of Delta. High Gamma near expiry means unstable hedge ratio |
| Vega (ν) | S·e^(-qT)·N'(d1)·√T / 100 | > 0 | P&L change per +1% in volatility. Long options always have positive Vega |
| Theta (Θ) | See full formula (negative, daily decay) | < 0 (long) | Daily P&L erosion from time alone. Accelerates near expiry (Theta burn) |
| Rho (ρ) | Call: K·T·e^(-rT)·N(d2)/100 / Put: -K·T·e^(-rT)·N(-d2)/100 | Call > 0 / Put < 0 | P&L change per +1% in risk-free rate. Typically small vs other Greeks |
""")

        st.markdown("### P&L Chart — How to Read It")
        st.markdown("""
- **Blue curve** — P&L at expiry as a function of the final spot price
- **Green zone** — scenarios where you make money at expiry
- **Red zone** — scenarios where you lose money at expiry
- **BE label** — exact break-even spot level (where the curve crosses zero)
- **K label** — strike: for a call, profit starts here; for a put, profit ends here
- **S label** — current spot: where the underlying is trading today

> The chart shows **expiry P&L only** — it does not reflect mark-to-market value before expiry, which would be the BS price curve, not the kinked payoff.
""")

        st.markdown("### Key Assumptions & Limitations")
        st.markdown("""
- Volatility is **constant** over the life of the option (not true in practice — see Vol Skew)
- **No early exercise** — only valid for European options
- **Log-normal** distribution of returns — underestimates fat tails / crash risk
- **Continuous** hedging with no transaction costs
- Risk-free rate is constant and known
""")

    # ── MONTE CARLO ───────────────────────────────────────────────────────────
    with doc_tab2:
        st.markdown("## Monte Carlo Simulation")
        st.markdown("""
Monte Carlo (MC) pricing simulates thousands of possible price paths for the underlying asset and averages the
discounted payoffs. It is more flexible than Black-Scholes: it can handle path-dependent payoffs, stochastic
volatility, and complex structures. Here it is applied to vanilla options as a validation and learning tool.
""")
        st.markdown("### Price Path Simulation (GBM)")
        st.markdown('<div class="formula-box">'
            '<b>Discretised GBM (Euler scheme):</b><br><br>'
            'S(t+dt) = S(t) · exp[ (r - q - σ²/2)·dt  +  σ·√dt·Z ]<br><br>'
            'where  Z ~ N(0,1)  (standard normal random draw)<br><br>'
            '<b>Terminal price after N steps:</b><br>'
            'S(T) = S(0) · exp[ Σ { (r-q-σ²/2)·dt + σ·√dt·Zᵢ } ]  for i=1..N'
            '</div>', unsafe_allow_html=True)

        st.markdown("### Option Price Estimate")
        st.markdown('<div class="formula-box">'
            '<b>Call payoff per path:</b>  max(S(T) - K, 0)<br>'
            '<b>Put payoff per path:</b>   max(K - S(T), 0)<br><br>'
            '<b>MC Price estimate:</b>  C ≈ e^(-rT) · (1/M) · Σ payoff(i)  for i=1..M paths<br><br>'
            '<b>Standard Error:</b>  SE = e^(-rT) · std(payoffs) / √M<br><br>'
            'As M → ∞, MC price → BS price (same model assumptions)'
            '</div>', unsafe_allow_html=True)

        st.markdown("### Variance Reduction — Antithetic Variates")
        st.markdown('<div class="formula-box">'
            'For each random draw Z, also simulate -Z.<br>'
            'This creates pairs of negatively correlated paths, reducing variance of the estimator<br>'
            'without increasing the number of model evaluations.<br><br>'
            'Effective sample size doubles with minimal computational overhead.'
            '</div>', unsafe_allow_html=True)

        st.markdown("### MC-Specific Settings")
        st.markdown("""
| Setting | Effect | Recommended |
|---|---|---|
| Simulations (M) | More paths → lower Standard Error → more accurate price. SE ∝ 1/√M | 100,000+ |
| Time steps (N) | More steps → finer discretisation of the path. 252 = daily steps over 1 year | 252 (daily) |
| Antithetic variates | Halves variance of the estimator. Always recommended unless benchmarking | On |
| Seed | Fixes the random number sequence for reproducibility | Any fixed integer |
""")

        st.markdown("### Output — Distribution Chart")
        st.markdown("""
The histogram displays the **distribution of simulated terminal prices S(T)** across all M paths.

- **Shape** — log-normal bell curve centred around the forward price F = S·e^((r-q)T)
- **Width** — increases with σ and T: higher volatility = fatter, wider distribution
- **K line** — shows how many paths finish in-the-money (right of K for calls, left for puts)
- **Useful insight** — visualises the probability mass in/out of the money and the tail risk
""")

    # ── IMPLIED VOLATILITY ────────────────────────────────────────────────────
    with doc_tab3:
        st.markdown("## Implied Volatility Calibration")
        st.markdown("""
Implied Volatility (IV) is the volatility value σ* that, when plugged into the Black-Scholes formula,
reproduces the observed market price of an option. It is the market's forward-looking estimate of
uncertainty — not a historical measure.
""")
        st.markdown("### Calibration Problem")
        st.markdown('<div class="formula-box">'
            '<b>Find σ* such that:</b><br><br>'
            'BS(S, K, T, r, σ*, q) = C_market<br><br>'
            'There is no closed-form inverse of BS with respect to σ.<br>'
            'This equation is solved numerically using <b>Brent\'s root-finding method</b>:<br>'
            '→ Bracket: σ ∈ [0.001, 5.0]  (0.1% to 500% vol)<br>'
            '→ Convergence: typically < 100 iterations<br>'
            '→ Returns NaN if no solution exists (e.g. market price < intrinsic value)'
            '</div>', unsafe_allow_html=True)

        st.markdown("### Output Metrics")
        st.markdown("""
| Metric | Meaning |
|---|---|
| Implied Vol | σ* — the volatility implied by the observed market price |
| Market price | The option price as traded/quoted in the market |
| Theoretical price | BS price recalculated using σ* — should equal market price (sanity check) |
| Vega | Sensitivity of option price to a +1% change in σ* — measures IV risk |
""")

        st.markdown("### Volatility Skew Chart")
        st.markdown("""
Plots IV against **moneyness** (K/S) for both calls and puts at the same maturity.

- **Flat line** — Black-Scholes world: constant IV across all strikes (never observed in practice)
- **Downward slope (left skew)** — typical for equity markets: OTM puts are expensive (crash insurance demand), OTM calls are cheap. This is called the **put skew** or **volatility smirk**
- **Smile** — IV is elevated for both OTM puts and OTM calls, common in FX and commodity markets
- **ATM line** — the calibrated IV for your input strike. All other strikes are rescaled from this point

> In this tool the skew is synthetic — it recalibrates IV for each strike using the same input σ. It illustrates the mechanics of the skew, not real market data.
""")

        st.markdown("### Term Structure Chart")
        st.markdown("""
Plots IV against **maturity** for a fixed strike.

- **Upward sloping (contango)** — typical in calm markets: longer-dated options carry more uncertainty premium
- **Inverted (backwardation)** — near-term IV is elevated vs long-term, signals short-term stress or upcoming event (earnings, election, central bank)
- **Flat** — market pricing uniform uncertainty across maturities

> Same synthetic rescaling applies here. The term structure reflects how volatility changes with time to expiry, holding the strike fixed.
""")

    # ── BACKTESTING ───────────────────────────────────────────────────────────
    with doc_tab4:
        st.markdown("## Strategy Backtesting")
        st.markdown("""
The backtester simulates M Monte Carlo price paths and computes the **realised P&L** of a given options strategy
on each path. It provides a statistical view of strategy performance: win rate, distribution of outcomes, tail risk.
""")
        st.markdown("### Path Generation")
        st.markdown('<div class="formula-box">'
            'Same GBM discretisation as Monte Carlo pricing.<br>'
            'M paths of length N days are simulated. Only the <b>terminal price S(T)</b> is used for P&L.<br><br>'
            'For each path:  P&L = Payoff at expiry - Entry cost at inception'
            '</div>', unsafe_allow_html=True)

        st.markdown("### Strategy P&L Formulas")
        st.markdown("""
| Strategy | P&L Formula | Max Gain | Max Loss | Breakeven |
|---|---|---|---|---|
| Long Call | max(S(T)-K, 0) - C₀ | Unlimited | Premium C₀ | K + C₀ |
| Long Put | max(K-S(T), 0) - P₀ | K - premium | Premium P₀ | K - P₀ |
| Covered Call | (S(T)-S₀) + C₀ - max(S(T)-K, 0) | C₀ (capped at K) | Stock → 0 | S₀ - C₀ |
| Protective Put | (S(T)-S₀) - P₀ + max(K-S(T), 0) | Unlimited | Premium P₀ | S₀ + P₀ |
| Straddle | max(S(T)-K,0) + max(K-S(T),0) - C₀ - P₀ | Unlimited | C₀ + P₀ | K ± (C₀+P₀) |
| Strangle | max(S(T)-Kc,0) + max(Kp-S(T),0) - Cc - Pp | Unlimited | Cc + Pp | Kc+premium or Kp-premium |
""")

        st.markdown("### Output Metrics")
        st.markdown("""
| Metric | Formula | Interpretation |
|---|---|---|
| Avg P&L | Mean of all path P&Ls | Expected value of the strategy under GBM — positive does not mean risk-free |
| Median P&L | 50th percentile of P&L distribution | More robust than mean when distribution is skewed |
| Win Rate | % of paths with P&L > 0 | How often the strategy expires in profit — high WR ≠ good strategy |
| Max Gain | max(P&L across all paths) | Best-case scenario across simulations |
| Max Loss | min(P&L across all paths) | Worst-case scenario — key metric for risk management |
| Sharpe | Avg P&L / Std(P&L) × √252 | Annualised risk-adjusted return. >1 = good, >2 = strong, <0 = destroys value |
""")

        st.markdown("### P&L Distribution Chart — How to Read It")
        st.markdown("""
- **Green bars** — scenarios where the strategy expired profitable
- **Red bars** — scenarios where the strategy expired at a loss
- **Avg line (blue dashed)** — expected P&L across all simulations
- **BE line (grey)** — P&L = 0; divides winners from losers
- **Shape of distribution** — skewed right = most losses are small, few large wins (e.g. long call). Skewed left = most wins are small, few large losses (e.g. covered call, short option)
""")

        st.markdown("### P&L vs Final Spot Chart — How to Read It")
        st.markdown("""
- **X-axis** — final spot price at expiry across all simulated paths
- **Y-axis** — realised P&L for that path
- **Green dots** — profitable outcomes; **purple dots** — loss outcomes
- **S0 line (yellow)** — starting spot, separates paths that moved up vs down
- **K line (grey)** — strike level: for a call, all green dots should be to the right of K (where S(T) > K)
- **BE line** — P&L = 0 horizontal; allows you to visually read the breakeven spot
- **Shape** — reveals the payoff profile visually. A covered call will show a flat cap at the top (profit capped at K); a long call will show a straight diagonal rising slope above K
""")

        st.markdown("### Percentile Table")
        st.markdown("""
| Percentile | Meaning |
|---|---|
| 5% | Worst 5% of outcomes — tail risk / near worst-case |
| 25% | Lower quartile — typical bad outcome |
| 50% | Median outcome — most representative central scenario |
| 75% | Upper quartile — typical good outcome |
| 95% | Best 5% of outcomes — near best-case |

> A strategy with a high Win Rate but a very negative 5th percentile is a **short-volatility profile** — you win often but when you lose, you lose big (e.g. short straddle equivalent).
""")

# ══════════════════════════════════════════════════════════════════════════════
    # ── DELTA HEDGING DOC ─────────────────────────────────────────────────────
    with doc_tab5:
        st.markdown("## Delta Hedging")
        st.markdown("""
Delta hedging is the core risk management technique used by options market makers.
The goal is to eliminate **directional risk (Delta)** by continuously adjusting a stock position,
so that the portfolio P&L depends only on **Gamma** and **Vega** — not on which way the market moves.
""")
        st.markdown("### The Concept")
        st.markdown('<div class="formula-box">'
            '<b>Long call position:</b>  Delta > 0  (gains when S rises)<br><br>'
            '<b>To hedge:</b>  Short Delta shares of stock<br>'
            '<b>Net portfolio Delta</b> = option Delta + stock Delta = Δ - Δ = 0<br><br>'
            '<b>Problem:</b>  Delta changes as S moves (Gamma effect)<br>'
            '<b>Solution:</b>  Rebalance the stock position continuously (or at discrete intervals)<br><br>'
            '<b>Cost of hedging:</b>  Each rebalance = buy high / sell low when Gamma > 0<br>'
            'This hedging cost = Theta decay of the option (no free lunch)'
            '</div>', unsafe_allow_html=True)

        st.markdown("### P&L Attribution of a Delta-Hedged Portfolio")
        st.markdown('<div class="formula-box">'
            '<b>Daily P&L of hedged portfolio (Taylor expansion):</b><br><br>'
            'P&L ≈  ½ · Γ · (ΔS)²  +  ν · Δσ  +  Θ · Δt<br><br>'
            'where:<br>'
            '  Γ · (ΔS)²/2  =  Gamma P&L  (realised vol benefit)<br>'
            '  ν · Δσ        =  Vega P&L   (change in implied vol)<br>'
            '  Θ · Δt        =  Theta decay (time cost — always negative for long options)<br><br>'
            '<b>Key insight:</b>  If realised vol > implied vol → Gamma P&L > Theta cost → hedging is profitable<br>'
            'If realised vol < implied vol → Gamma P&L < Theta cost → hedging loses money'
            '</div>', unsafe_allow_html=True)

        st.markdown("### Simulation Logic")
        st.markdown("""
| Step | Action |
|---|---|
| t = 0 | Buy option for C₀. Set stock position = -Δ₀ (short Delta shares). Cash = -C₀ + Δ₀·S₀ |
| t = 1..N-1 | At each rebalancing step: compute new Δₜ, adjust stock position, debit/credit cash. Accrue risk-free interest on cash. |
| t = N | Receive option payoff. Unwind stock at Sₙ. Final P&L = cash + stock value + payoff |
""")

        st.markdown("### Output Metrics")
        st.markdown("""
| Metric | Meaning |
|---|---|
| Avg Hedged P&L | Should be near zero if vol assumptions hold. Positive if realised vol > implied vol. |
| Avg Unhedged P&L | Raw directional P&L without any hedge |
| Hedge Std Dev | Dispersion of hedged P&L — lower = better hedge quality |
| Hedge Win Rate | % of paths where hedging generated positive P&L |
| Entry Cost | BS price paid at inception |
""")

        st.markdown("### Charts")
        st.markdown("""
**P&L Distribution — Hedged vs Unhedged**
- Purple = unhedged P&L distribution (wide, directional)
- Cyan = delta-hedged P&L distribution (should be narrow, centred near 0)
- The tighter the hedged distribution, the more effective the hedge
- Residual spread = hedging error from discrete rebalancing + vol mismatch

**Delta Evolution Over Time**
- Each cyan line = Delta path on one simulated price path
- Yellow line = average Delta across all paths
- Delta starts near 0.5 for ATM options, drifts toward 0 (OTM) or 1 (ITM) as time passes
- Near expiry: Delta collapses rapidly — Gamma spikes — hedging becomes very costly
""")

        st.markdown("### Rebalancing Frequency — Practical Implications")
        st.markdown("""
| Frequency | Hedging Error | Transaction Cost | Used by |
|---|---|---|---|
| Continuous | Zero (theoretical) | Infinite | Textbooks only |
| Daily | Very low | Low-moderate | Equity options desks |
| Weekly | Moderate | Low | Some structured product desks |
| At expiry | Maximum (= unhedged) | None | No hedge |

> In practice, desks hedge based on **Delta bands** (e.g. rebalance when Delta moves by 0.05)
> rather than fixed time intervals — this balances hedging error against transaction costs.
""")

        st.markdown("### Why This Matters")
        st.markdown("""
- Market makers **buy/sell options** and immediately delta-hedge to isolate Vega and Gamma exposure
- The P&L of a market maker comes from the **spread between implied vol (sold) and realised vol (hedged)**
- A trader long Gamma wants **large spot moves** (high realised vol) to offset Theta decay
- A trader short Gamma (e.g. sold a straddle) wants **low realised vol** — spot stays still
- Understanding this is fundamental for any derivatives trading or structuring role
""")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: APP
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "app":

    title_map = {"Pricing": f"Options Pricer — {pricing_method}",
                 "Implied Volatility": "Implied Volatility Calibrator",
                 "Backtesting": "Strategy Backtester"}
    st.markdown(f"# {title_map[mode]}")
    st.markdown('<div class="author-link">by <a href="https://www.linkedin.com/in/arthurcotten/">Arthur Cotten</a> • <a href="https://github.com/arthurcotten">@arthurcotten</a></div>', unsafe_allow_html=True)
    st.markdown("---")

    T = T_day / 365

    # ── PRICING ───────────────────────────────────────────────────────────────
    if mode == "Pricing":
        if pricing_method == "Black-Scholes":
            price=bs(S,K,T,r,sigma,q,opt); g=greeks(S,K,T,r,sigma,q,opt)
            std_error=None; mc_paths=None
        else:
            with st.spinner("Running Monte Carlo..."):
                try:
                    mc=monte_carlo_pricer_cached(S,K,T,r,sigma,q,opt,n_sims,n_steps,antithetic,int(seed))
                    price=mc["price"]; std_error=mc["std_error"]; mc_paths=mc["paths"]
                    g={k:mc[k] for k in ["delta","gamma","vega","theta","rho"]}
                except Exception as e:
                    st.error(f"Monte Carlo error: {e}"); st.stop()

        prob=prob_itm(S,K,T,r,sigma,q,opt)
        cost=prem if prem>0 else price
        be=(K+cost) if opt=="call" else (K-cost)
        intr=max(S-K,0) if opt=="call" else max(K-S,0); tv=price-intr
        mon=S/K; mlbl="ATM" if abs(mon-1)<0.01 else ("ITM" if (opt=="call" and mon>1) or (opt=="put" and mon<1) else "OTM")

        c1,c2,c3,c4,c5,c6=st.columns(6)
        c1.metric("Price",      f"${price:.4f}")
        c2.metric("Std Error" if std_error else "Break-even", f"${std_error:.4f}" if std_error else f"${be:.2f}")
        c3.metric("Prob ITM",   f"{prob*100:.1f}%")
        c4.metric("Time Value", f"${tv:.4f}")
        c5.metric("Moneyness",  mlbl)
        c6.metric("Intrinsic",  f"${intr:.4f}")

        st.markdown("---"); st.markdown("### Greeks")
        g1,g2,g3,g4,g5=st.columns(5)
        g1.metric("Delta",f"{g['delta']:+.5f}"); g2.metric("Gamma",f"{g['gamma']:.6f}")
        g3.metric("Vega", f"{g['vega']:.5f}");  g4.metric("Theta",f"{g['theta']:+.5f}")
        g5.metric("Rho",  f"{g['rho']:+.5f}")

        st.markdown("---")
        alr=[]
        if T<7/365:              alr.append("⚠️ Maturity < 7 days")
        if abs(g["delta"])<0.10: alr.append("⚠️ Delta very low")
        if tv<0.005:             alr.append("⚠️ Time value near zero")
        if prob<0.15:            alr.append("⚠️ Prob ITM < 15%")
        for a in alr: st.warning(a)
        if not alr: st.success("✓ OK")

        Sr=np.linspace(S*0.7,S*1.3,300)
        col1,col2=st.columns([2,1])
        with col1:
            fig,ax=plt.subplots(figsize=(4.2,2.4),facecolor=BG); ax.set_facecolor(PANEL)
            pnl=(np.maximum(Sr-K,0)-cost if opt=="call" else np.maximum(K-Sr,0)-cost)
            ym,yM=pnl.min(),pnl.max(); yp=(yM-ym)*0.12
            ax.fill_between(Sr,pnl,0,where=pnl>=0,alpha=0.12,color=GREEN,zorder=1)
            ax.fill_between(Sr,pnl,0,where=pnl<0, alpha=0.12,color=RED,  zorder=1)
            ax.plot(Sr,pnl,color=ACCENT,lw=1.2,zorder=3)
            ax.axhline(0,color=GRAY,lw=0.5,alpha=0.5,zorder=2)
            vline(ax,K, f"K  ${K:.0f}",   YELLOW,    ym-yp,yM)
            vline(ax,be,f"BE  ${be:.2f}", GREEN,     ym-yp,yM)
            vline(ax,S, f"S  ${S:.0f}",   "#9ca3af", ym-yp,yM)
            ax.set_ylim(ym-yp*1.8,yM+yp); annotate_be(ax,be,ym-yp*1.6)
            sty(ax,f"P&L  ·  {opt.upper()}","Spot ($)","P&L ($)")
            fig.tight_layout(pad=1.2); st.pyplot(fig,use_container_width=True); plt.close(fig)
        with col2:
            if pricing_method=="Monte Carlo" and mc_paths is not None:
                fig2,ax2=plt.subplots(figsize=(3.1,3.2),facecolor=BG); ax2.set_facecolor(PANEL)
                ax2.hist(mc_paths,bins=34,color=CYAN,alpha=0.6,edgecolor="none")
                yh=ax2.get_ylim()[1]; vline(ax2,K,f"K  ${K:.0f}",YELLOW,0,yh)
                sty(ax2,"Distribution  S(T)","Terminal price ($)","Freq")
                fig2.tight_layout(pad=1.2); st.pyplot(fig2,use_container_width=True); plt.close(fig2)

    # ── IMPLIED VOLATILITY ────────────────────────────────────────────────────
    elif mode == "Implied Volatility":
        with st.spinner("Calibrating..."):
            iv=implied_volatility(market_price,S,K,T,r,q,opt)
        if np.isnan(iv):
            st.error("❌ Cannot calibrate IV — check inputs")
        else:
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Implied Vol",        f"{iv*100:.2f}%")
            c2.metric("Market price",       f"${market_price:.4f}")
            c3.metric("Theoretical price",  f"${bs(S,K,T,r,iv,q,opt):.4f}")
            c4.metric("Vega",               f"{greeks(S,K,T,r,iv,q,opt)['vega']:.5f}")
            st.markdown("---"); st.markdown("### Volatility Surface")
            cs,ct=st.columns(2)
            with cs:
                stks=np.linspace(S*0.7,S*1.3,20)
                ivc=[implied_volatility(bs(S,k,T,r,iv,q,"call"),S,k,T,r,q,"call") for k in stks]
                ivp=[implied_volatility(bs(S,k,T,r,iv,q,"put"), S,k,T,r,q,"put")  for k in stks]
                fig_s,ax=plt.subplots(figsize=(4.8,3.4),facecolor=BG); ax.set_facecolor(PANEL)
                vc=[(k/S,v*100) for k,v in zip(stks,ivc) if v and not np.isnan(v)]
                vp_=[(k/S,v*100) for k,v in zip(stks,ivp) if v and not np.isnan(v)]
                if vc: xc,yc=zip(*vc); ax.plot(xc,yc,color=CYAN,  lw=1.1,marker='o',markersize=3.2,label='Calls',markeredgewidth=0)
                if vp_: xp,yp=zip(*vp_); ax.plot(xp,yp,color=PURPLE,lw=1.1,marker='s',markersize=3.2,label='Puts', markeredgewidth=0)
                ax.axvline(1.0,   color=GRAY,  lw=0.5,linestyle=":",alpha=0.6,label="ATM")
                ax.axhline(iv*100,color=ACCENT,lw=0.5,linestyle="--",alpha=0.6)
                ax.legend(fontsize=7,facecolor=PANEL,edgecolor="#2a4a6b",labelcolor=TEXT,framealpha=0.8)
                sty(ax,"Skew  ·  Calls vs Puts","Moneyness (K/S)","IV (%)")
                fig_s.tight_layout(pad=1.2); st.pyplot(fig_s,use_container_width=True); plt.close(fig_s)
            with ct:
                mats=np.linspace(max(T,7/365),min(T*3,1.0),12)
                ivt=[implied_volatility(bs(S,K,m,r,iv,q,"call"),S,K,m,r,q,"call") for m in mats]
                fig_t,ax=plt.subplots(figsize=(4.8,3.4),facecolor=BG); ax.set_facecolor(PANEL)
                vt=[(m*365,v*100) for m,v in zip(mats,ivt) if v and not np.isnan(v)]
                if vt: xt,yt=zip(*vt); ax.plot(xt,yt,color=CYAN,lw=1.1,marker='o',markersize=3.2,markeredgewidth=0)
                ax.axvline(T*365, color=GRAY,  lw=0.5,linestyle=":",alpha=0.6)
                ax.axhline(iv*100,color=ACCENT,lw=0.5,linestyle="--",alpha=0.6)
                sty(ax,"Term Structure","Maturity (days)","IV (%)")
                fig_t.tight_layout(pad=1.2); st.pyplot(fig_t,use_container_width=True); plt.close(fig_t)

    # ── BACKTESTING ───────────────────────────────────────────────────────────
    elif mode == "Backtesting":

        # ── INTERNAL VALIDATORS ───────────────────────────────────────────────
        val_errors, val_warnings, val_ok = [], [], []

        # 1. Put-call parity check
        call_p = bs(S, K, T, r, sigma, q, "call")
        put_p  = bs(S, K, T, r, sigma, q, "put")
        pcp_lhs = call_p - put_p
        pcp_rhs = S * np.exp(-q*T) - K * np.exp(-r*T)
        pcp_err = abs(pcp_lhs - pcp_rhs)
        if pcp_err < 1e-6:
            val_ok.append(f"Put-Call Parity ✓  (error = {pcp_err:.2e})")
        else:
            val_errors.append(f"Put-Call Parity violation: error = {pcp_err:.4f}")

        # 2. Delta bounds
        g_check = greeks(S, K, T, r, sigma, q, "call")
        if 0 <= g_check["delta"] <= 1:
            val_ok.append(f"Call Delta in [0,1] ✓  ({g_check['delta']:.4f})")
        else:
            val_errors.append(f"Call Delta out of bounds: {g_check['delta']:.4f}")

        # 3. Option price >= intrinsic value
        intr_check = max(S - K, 0)
        if call_p >= intr_check - 1e-6:
            val_ok.append(f"Call price >= intrinsic ✓  (${call_p:.4f} >= ${intr_check:.4f})")
        else:
            val_errors.append(f"Call price below intrinsic: ${call_p:.4f} < ${intr_check:.4f}")

        # 4. Time value should be positive when OTM/ATM
        tv_check = call_p - intr_check
        if tv_check >= 0:
            val_ok.append(f"Time value >= 0 ✓  (${tv_check:.4f})")
        else:
            val_errors.append(f"Negative time value: ${tv_check:.4f}")

        # 5. Gamma > 0
        if g_check["gamma"] > 0:
            val_ok.append(f"Gamma > 0 ✓  ({g_check['gamma']:.6f})")
        else:
            val_errors.append(f"Gamma <= 0: {g_check['gamma']:.6f}")

        # 6. Theta < 0 for long option
        if g_check["theta"] < 0:
            val_ok.append(f"Theta < 0 ✓  ({g_check['theta']:.5f}  /day)")
        else:
            val_warnings.append(f"Theta positive for long option: {g_check['theta']:.5f}")

        # 7. Input sanity
        if sigma <= 0:
            val_errors.append("Volatility must be > 0")
        elif sigma > 2.0:
            val_warnings.append(f"Very high volatility: {sigma*100:.0f}% — results may be unreliable")
        if T <= 0:
            val_errors.append("Maturity must be > 0")
        if S <= 0 or K <= 0:
            val_errors.append("Spot and Strike must be > 0")

        # Display validators
        with st.expander("✅ Internal Validators", expanded=False):
            if val_errors:
                for e in val_errors:
                    st.error(f"❌  {e}")
            if val_warnings:
                for w in val_warnings:
                    st.warning(f"⚠️  {w}")
            for o in val_ok:
                st.success(f"✓  {o}")
            st.markdown(f"""
**Summary:**  {len(val_ok)} checks passed · {len(val_warnings)} warnings · {len(val_errors)} errors

| Check | Description |
|---|---|
| Put-Call Parity | C - P = S·e^(-qT) - K·e^(-rT) — fundamental no-arbitrage identity |
| Delta bounds | Call Delta must be in [0,1], Put Delta in [-1,0] |
| Price >= intrinsic | Option can never be worth less than its exercise value |
| Time value >= 0 | Intrinsic is the floor — any premium above it is time value |
| Gamma > 0 | Long options always have positive Gamma |
| Theta < 0 | Long options lose value with time (all else equal) |
| Input ranges | Volatility, spot, strike, maturity must be economically valid |
""")

        if val_errors:
            st.error("❌ Critical validation errors detected — results may be incorrect.")
            st.stop()

        # ── RUN BACKTEST ──────────────────────────────────────────────────────
        with st.spinner("Running backtest..."):
            try:
                df = backtest_strategy_cached(strategy,S,K,T,r,sigma,q,backtest_days,n_simulations)
            except Exception as e:
                st.error(f"Error: {e}"); st.stop()

        mp=df['pnl'].mean(); med=df['pnl'].median(); sdp=df['pnl'].std()
        wr=(df['pnl']>0).sum()/len(df)*100; mg=df['pnl'].max(); ml=df['pnl'].min()
        shr=(mp/sdp*np.sqrt(252)) if sdp>0 else 0

        st.markdown("### Performance")
        c1,c2,c3,c4,c5,c6=st.columns(6)
        c1.metric("Avg P&L",   f"${mp:.2f}"); c2.metric("Median P&L",f"${med:.2f}")
        c3.metric("Win Rate",  f"{wr:.1f}%"); c4.metric("Max Gain",  f"${mg:.2f}")
        c5.metric("Max Loss",  f"${ml:.2f}"); c6.metric("Sharpe",    f"{shr:.3f}")

        st.markdown("---")
        ch,cs2=st.columns(2)
        with ch:
            fig_h,ax=plt.subplots(figsize=(4.8,3.4),facecolor=BG); ax.set_facecolor(PANEL)
            pa=df['pnl'].values; bns=np.linspace(pa.min(),pa.max(),44)
            ax.hist(pa[pa>=0],bins=bns,color=GREEN,alpha=0.55,edgecolor="none")
            ax.hist(pa[pa<0], bins=bns,color=RED,  alpha=0.55,edgecolor="none")
            yh2=ax.get_ylim()[1]
            ax.axvline(mp,color=ACCENT,lw=0.8,linestyle="--",alpha=0.9,label=f"Avg  ${mp:.2f}")
            ax.axvline(0, color=GRAY,  lw=0.5,alpha=0.6,label="BE  $0.00")
            ax.set_ylim(0,yh2*1.12)
            ax.annotate("BE $0.00",xy=(0,yh2*0.02),fontsize=4.5,color=GRAY,fontfamily="monospace",ha="center",va="bottom",alpha=0.8)
            ax.legend(fontsize=7,facecolor=PANEL,edgecolor="#2a4a6b",labelcolor=TEXT,framealpha=0.8)
            sty(ax,f"P&L Distribution  ·  {strategy.replace('_',' ').title()}","P&L ($)","Freq")
            fig_h.tight_layout(pad=1.2); st.pyplot(fig_h,use_container_width=True); plt.close(fig_h)
        with cs2:
            fig_sc,ax=plt.subplots(figsize=(4.8,3.4),facecolor=BG); ax.set_facecolor(PANEL)
            sp=df['final_spot'].values; pn=df['pnl'].values
            ax.scatter(sp[pn>=0],pn[pn>=0],alpha=0.45,s=8,color=GREEN, edgecolors="none",zorder=3)
            ax.scatter(sp[pn<0], pn[pn<0], alpha=0.45,s=8,color=PURPLE,edgecolors="none",zorder=3)
            ys,yS=pn.min(),pn.max(); yps=(yS-ys)*0.12
            ax.axhline(0,color=GRAY,lw=0.5,alpha=0.55,zorder=2)
            vline(ax,S,f"S0  ${S:.0f}",YELLOW,   ys-yps,yS)
            vline(ax,K,f"K  ${K:.0f}", "#9ca3af",ys-yps,yS)
            ax.set_ylim(ys-yps*1.8,yS+yps)
            ax.annotate("BE $0.00",xy=(sp.min()+(sp.max()-sp.min())*0.03,0),fontsize=4.5,color=GRAY,fontfamily="monospace",ha="left",va="bottom",alpha=0.8)
            ax.legend(handles=[
                plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=GREEN, markersize=5,label='Gain',linewidth=0),
                plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=PURPLE,markersize=5,label='Loss',linewidth=0),
                plt.Line2D([0],[0],color=YELLOW,lw=0.8,linestyle='--',label='S0'),
            ],fontsize=7,facecolor=PANEL,edgecolor="#2a4a6b",labelcolor=TEXT,framealpha=0.8)
            sty(ax,"P&L vs Final Spot","Final spot ($)","P&L ($)")
            fig_sc.tight_layout(pad=1.2); st.pyplot(fig_sc,use_container_width=True); plt.close(fig_sc)

        st.markdown("---")
        st.markdown("### Percentiles")
        pcts=[5,25,50,75,95]
        st.dataframe(pd.DataFrame({
            "Percentile": [f"{p}%" for p in pcts],
            "P&L ($)":    [f"${df['pnl'].quantile(p/100):.2f}" for p in pcts],
            "Return (%)": [f"{df['return_pct'].quantile(p/100):.2f}%" for p in pcts],
        }),use_container_width=True,hide_index=True)

        # ── DELTA HEDGING ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Delta Hedging Simulation")
        st.markdown(f"*Rebalancing: **{hedge_freq}** — {hedge_n_paths} paths  ·  strategy: long {opt}*")

        with st.spinner("Running delta hedge simulation..."):
            hres = delta_hedge_simulation(S, K, T, r, sigma, q, opt,
                                          backtest_days, hedge_n_paths, hedge_freq)

        hp  = hres["hedge_pnls"]
        uhp = hres["unhedged_pnls"]
        dps = hres["delta_paths"]

        # Hedge validator
        hedge_val_ok = []
        if abs(hp.mean()) < abs(uhp.mean()) * 0.5:
            hedge_val_ok.append(f"Hedge reduces avg P&L magnitude ✓  (hedged: ${hp.mean():.3f}  vs  unhedged: ${uhp.mean():.2f})")
        if hp.std() < uhp.std():
            hedge_val_ok.append(f"Hedge reduces P&L dispersion ✓  (std: ${hp.std():.3f}  vs  ${uhp.std():.2f})")
        entry = hres["entry_cost"]
        bs_check = bs(S, K, T, r, sigma, q, opt)
        if abs(entry - bs_check) < 1e-4:
            hedge_val_ok.append(f"Entry cost matches BS price ✓  (${entry:.4f})")

        if hedge_val_ok:
            with st.expander("✅ Hedge Validators", expanded=False):
                for v in hedge_val_ok:
                    st.success(f"✓  {v}")

        hc1,hc2,hc3,hc4,hc5 = st.columns(5)
        hc1.metric("Avg Hedged P&L",   f"${hp.mean():.3f}")
        hc2.metric("Avg Unhedged P&L", f"${uhp.mean():.2f}")
        hc3.metric("Hedge Std Dev",    f"${hp.std():.3f}")
        hc4.metric("Hedge Win Rate",   f"{(hp>0).sum()/len(hp)*100:.1f}%")
        hc5.metric("Entry Cost",       f"${hres['entry_cost']:.4f}")

        st.markdown("""
> **Key insight:** A delta-hedged P&L near zero confirms the model is consistent.
> Positive residual = realised vol exceeded implied vol (Gamma won over Theta).
> The wider the hedged distribution vs a flat line at zero, the greater the hedging error from discrete rebalancing.
""")

        col_hh, col_hs = st.columns(2)
        with col_hh:
            fig_hh,ax=plt.subplots(figsize=(4.2,2.8),facecolor=BG); ax.set_facecolor(PANEL)
            all_vals = np.concatenate([hp, uhp])
            bins_h   = np.linspace(all_vals.min(), all_vals.max(), 40)
            ax.hist(uhp, bins=bins_h, color=PURPLE, alpha=0.5, edgecolor="none", label="Unhedged")
            ax.hist(hp,  bins=bins_h, color=CYAN,   alpha=0.6, edgecolor="none", label="Delta-Hedged")
            ax.axvline(0,        color=GRAY,  lw=0.6, linestyle="-",  alpha=0.7)
            ax.axvline(hp.mean(),color=CYAN,  lw=0.9, linestyle="--", alpha=0.9, label=f"Hedged avg ${hp.mean():.3f}")
            ax.legend(fontsize=6.5, facecolor=PANEL, edgecolor="#2a4a6b", labelcolor=TEXT, framealpha=0.8)
            sty(ax,"P&L  ·  Hedged vs Unhedged","P&L ($)","Freq")
            fig_hh.tight_layout(pad=1.2); st.pyplot(fig_hh,use_container_width=True); plt.close(fig_hh)

        with col_hs:
            fig_hs,ax=plt.subplots(figsize=(4.2,2.8),facecolor=BG); ax.set_facecolor(PANEL)
            n_show = min(25, len(dps))
            days_x = np.arange(backtest_days)
            for dp in dps[:n_show]:
                length = min(len(dp), backtest_days)
                ax.plot(days_x[:length], dp[:length], color=CYAN, lw=0.4, alpha=0.3)
            max_len = max(len(dp) for dp in dps[:n_show])
            mean_d  = np.array([
                np.mean([dps[i][t] for i in range(n_show) if t < len(dps[i])])
                for t in range(max_len)
            ])
            ax.plot(days_x[:len(mean_d)], mean_d, color=YELLOW, lw=1.2, label="Avg Delta")
            ax.axhline(0.5, color=GRAY, lw=0.5, linestyle=":", alpha=0.6, label="Δ = 0.5 (ATM)")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=6.5, facecolor=PANEL, edgecolor="#2a4a6b", labelcolor=TEXT, framealpha=0.8)
            sty(ax,"Delta Evolution Over Time","Day","Delta")
            fig_hs.tight_layout(pad=1.2); st.pyplot(fig_hs,use_container_width=True); plt.close(fig_hs)

"""
Options Pricer — Streamlit
Compatible GitHub Codespaces / navigateur
Lancer avec : streamlit run streamlit_bs_pricer.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

st.markdown("""
<style>
    .main { background-color: #0a0e17; }
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebar"] { background-color: #0f1219; }
    h1, h2, h3 { color: #4ade80; font-family: monospace; font-weight: normal; }
    p, span, div { color: #e5e7eb; }
    .metric-label { color: #4ade80 !important; font-size: 0.8rem !important; font-weight: normal !important; }
    .stMetric { background-color: #1a1f2e; border-radius: 8px; padding: 8px; border: 1px solid #22c55e; }
    div[data-testid="metric-container"] {
        background-color: #1a1f2e;
        border: 1px solid #22c55e;
        border-radius: 8px;
        padding: 10px;
    }
    div[data-testid="metric-container"] label {
        color: #4ade80 !important;
        font-weight: normal !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
    }
    .stMarkdown { color: #e5e7eb; }
    .stAlert { background-color: #1a1f2e; border: 1px solid #22c55e; }
    .author-link { 
        color: #9ca3af; 
        font-size: 0.85rem; 
        font-family: monospace; 
        margin-top: -10px;
        margin-bottom: 15px;
    }
    .author-link a {
        color: #4ade80;
        text-decoration: none;
        transition: color 0.2s;
    }
    .author-link a:hover {
        color: #22c55e;
        text-decoration: underline;
    }
    .dataframe {
        font-size: 0.85rem !important;
        font-family: monospace !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── STYLE MATPLOTLIB ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1219",
    "axes.facecolor":   "#1a1f2e",
    "axes.edgecolor":   "#22c55e",
    "axes.labelcolor":  "#4ade80",
    "text.color":       "#e5e7eb",
    "xtick.color":      "#e5e7eb",
    "ytick.color":      "#e5e7eb",
    "grid.color":       "#374151",
    "grid.linewidth":   0.6,
    "font.family":      "monospace",
})

BG     = "#0f1219"
PANEL  = "#1a1f2e"
BORDER = "#22c55e"
ACCENT = "#4ade80"
GREEN  = "#10b981"
RED    = "#ef4444"
YELLOW = "#f59e0b"
PURPLE = "#a78bfa"
CYAN   = "#06b6d4"
ORANGE = "#fb923c"
GRAY   = "#9ca3af"
TEXT   = "#e5e7eb"
TITLE  = "#4ade80"

# ─── BLACK-SCHOLES ────────────────────────────────────────────────────────────

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
    d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    nd1 = norm.pdf(d1)
    if opt == "call":
        delta = np.exp(-q*T)*norm.cdf(d1)
        theta = (-(S*np.exp(-q*T)*nd1*sigma)/(2*np.sqrt(T))
                 - r*K*np.exp(-r*T)*norm.cdf(d2)
                 + q*S*np.exp(-q*T)*norm.cdf(d1)) / 365
        rho = K*T*np.exp(-r*T)*norm.cdf(d2) / 100
    else:
        delta = -np.exp(-q*T)*norm.cdf(-d1)
        theta = (-(S*np.exp(-q*T)*nd1*sigma)/(2*np.sqrt(T))
                 + r*K*np.exp(-r*T)*norm.cdf(-d2)
                 - q*S*np.exp(-q*T)*norm.cdf(-d1)) / 365
        rho = -K*T*np.exp(-r*T)*norm.cdf(-d2) / 100
    gamma = np.exp(-q*T)*nd1 / (S*sigma*np.sqrt(T))
    vega  = S*np.exp(-q*T)*nd1*np.sqrt(T) / 100
    return {"delta":delta, "gamma":gamma, "vega":vega, "theta":theta, "rho":rho}

def prob_itm(S, K, T, r, sigma, q=0.0, opt="call"):
    if T <= 1e-10 or sigma <= 1e-10:
        return 1.0 if (opt=="call" and S>K) or (opt=="put" and S<K) else 0.0
    d2 = (np.log(S/K)+(r-q-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.cdf(d2) if opt=="call" else norm.cdf(-d2)

# ─── IMPLIED VOLATILITY ───────────────────────────────────────────────────────

def implied_volatility(market_price, S, K, T, r, q=0.0, opt="call"):
    """
    Calibration de la volatilité implicite par méthode de Brent
    """
    if T <= 1e-10:
        return np.nan
    
    # Vérification arbitrage
    intrinsic = max(S-K, 0) if opt=="call" else max(K-S, 0)
    if market_price < intrinsic:
        return np.nan
    
    def objective(sigma):
        try:
            return bs(S, K, T, r, sigma, q, opt) - market_price
        except:
            return 1e10
    
    try:
        iv = brentq(objective, 0.001, 5.0, maxiter=100)
        return iv
    except:
        return np.nan

# ─── MONTE CARLO ──────────────────────────────────────────────────────────────

def monte_carlo_pricer(S, K, T, r, sigma, q=0.0, opt="call", n_sims=100000, n_steps=252, antithetic=True, seed=42):
    """
    Monte Carlo pour options européennes avec variance reduction
    """
    np.random.seed(seed)
    dt = T / n_steps
    n_paths = n_sims // 2 if antithetic else n_sims
    
    Z = np.random.standard_normal((n_paths, n_steps))
    if antithetic:
        Z = np.concatenate([Z, -Z], axis=0)
    
    drift = (r - q - 0.5*sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    log_returns = drift + diffusion * Z
    log_price_paths = np.log(S) + np.cumsum(log_returns, axis=1)
    S_T = np.exp(log_price_paths[:, -1])
    
    if opt == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)
    
    price = np.exp(-r*T) * np.mean(payoffs)
    std_error = np.exp(-r*T) * np.std(payoffs) / np.sqrt(len(payoffs))
    
    # Greeks par différences finies
    dS = S * 0.01
    price_up = monte_carlo_price_only(S+dS, K, T, r, sigma, q, opt, n_sims//2, n_steps, antithetic, seed)
    price_down = monte_carlo_price_only(S-dS, K, T, r, sigma, q, opt, n_sims//2, n_steps, antithetic, seed)
    delta = (price_up - price_down) / (2*dS)
    gamma = (price_up - 2*price + price_down) / (dS**2)
    
    dsigma = sigma * 0.01
    price_vol_up = monte_carlo_price_only(S, K, T, r, sigma+dsigma, q, opt, n_sims//2, n_steps, antithetic, seed)
    vega = (price_vol_up - price) / dsigma / 100
    
    dT = 1/365
    if T > dT:
        price_t_down = monte_carlo_price_only(S, K, T-dT, r, sigma, q, opt, n_sims//2, n_steps, antithetic, seed)
        theta = (price_t_down - price) / dT / 365
    else:
        theta = 0.0
    
    dr = 0.001
    price_r_up = monte_carlo_price_only(S, K, T, r+dr, sigma, q, opt, n_sims//2, n_steps, antithetic, seed)
    rho = (price_r_up - price) / dr / 100
    
    return {
        "price": price,
        "std_error": std_error,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "paths": S_T[:min(1000, len(S_T))]
    }

def monte_carlo_price_only(S, K, T, r, sigma, q, opt, n_sims, n_steps, antithetic, seed):
    """Version allégée pour calcul de Greeks"""
    np.random.seed(seed)
    dt = T / n_steps
    n_paths = n_sims // 2 if antithetic else n_sims
    Z = np.random.standard_normal((n_paths, n_steps))
    if antithetic:
        Z = np.concatenate([Z, -Z], axis=0)
    drift = (r - q - 0.5*sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    log_returns = drift + diffusion * Z
    log_price_paths = np.log(S) + np.cumsum(log_returns, axis=1)
    S_T = np.exp(log_price_paths[:, -1])
    if opt == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)
    return np.exp(-r*T) * np.mean(payoffs)

# ─── BACKTESTING ──────────────────────────────────────────────────────────────

def backtest_strategy(strategy, S0, K, T, r, sigma, q, initial_capital, n_days, n_sims=1000):
    """
    Backtest d'une stratégie d'options
    strategy: "long_call", "long_put", "covered_call", "protective_put", "straddle", "strangle"
    """
    np.random.seed(42)
    dt = 1/252
    results = []
    
    for sim in range(n_sims):
        capital = initial_capital
        S = S0
        time_to_expiry = T
        
        # Position initiale
        if strategy == "long_call":
            entry_price = bs(S, K, time_to_expiry, r, sigma, q, "call")
            position = {"type": "call", "quantity": 1, "entry": entry_price}
            capital -= entry_price
            
        elif strategy == "long_put":
            entry_price = bs(S, K, time_to_expiry, r, sigma, q, "put")
            position = {"type": "put", "quantity": 1, "entry": entry_price}
            capital -= entry_price
            
        elif strategy == "covered_call":
            call_price = bs(S, K, time_to_expiry, r, sigma, q, "call")
            position = {"stock": 1, "call_short": 1, "call_entry": call_price}
            capital = capital - S + call_price
            
        elif strategy == "protective_put":
            put_price = bs(S, K, time_to_expiry, r, sigma, q, "put")
            position = {"stock": 1, "put_long": 1, "put_entry": put_price}
            capital = capital - S - put_price
            
        elif strategy == "straddle":
            call_price = bs(S, K, time_to_expiry, r, sigma, q, "call")
            put_price = bs(S, K, time_to_expiry, r, sigma, q, "put")
            position = {"call": 1, "put": 1, "call_entry": call_price, "put_entry": put_price}
            capital -= (call_price + put_price)
            
        elif strategy == "strangle":
            K_call = K * 1.05
            K_put = K * 0.95
            call_price = bs(S, K_call, time_to_expiry, r, sigma, q, "call")
            put_price = bs(S, K_put, time_to_expiry, r, sigma, q, "put")
            position = {"call": 1, "put": 1, "K_call": K_call, "K_put": K_put, 
                       "call_entry": call_price, "put_entry": put_price}
            capital -= (call_price + put_price)
        
        # Simulation du sous-jacent
        for day in range(n_days):
            Z = np.random.standard_normal()
            S = S * np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
            time_to_expiry -= dt
            
            if time_to_expiry <= 0:
                break
        
        # Calcul P&L à l'expiration
        if strategy == "long_call":
            payoff = max(S - K, 0)
            pnl = capital + payoff
            
        elif strategy == "long_put":
            payoff = max(K - S, 0)
            pnl = capital + payoff
            
        elif strategy == "covered_call":
            stock_value = S
            call_payoff = -max(S - K, 0)
            pnl = capital + stock_value + call_payoff
            
        elif strategy == "protective_put":
            stock_value = S
            put_payoff = max(K - S, 0)
            pnl = capital + stock_value + put_payoff
            
        elif strategy == "straddle":
            call_payoff = max(S - K, 0)
            put_payoff = max(K - S, 0)
            pnl = capital + call_payoff + put_payoff
            
        elif strategy == "strangle":
            call_payoff = max(S - position["K_call"], 0)
            put_payoff = max(position["K_put"] - S, 0)
            pnl = capital + call_payoff + put_payoff
        
        results.append({
            "final_spot": S,
            "pnl": pnl - initial_capital,
            "return_pct": (pnl - initial_capital) / initial_capital * 100
        })
    
    df = pd.DataFrame(results)
    return df

# ─── HELPERS PLOT ─────────────────────────────────────────────────────────────

def sty(ax, title, xl, yl):
    ax.set_title(title, color=TITLE, fontsize=9, pad=8, fontweight="normal")
    ax.set_xlabel(xl, color=TITLE, fontsize=8, fontweight="normal")
    ax.set_ylabel(yl, color=TITLE, fontsize=8, fontweight="normal")
    ax.grid(True, alpha=0.4, linewidth=0.7)
    ax.tick_params(labelsize=7.5, colors=TEXT, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ◈ OPTIONS PRICER")
    st.markdown("---")
    st.markdown("### Mode")
    
    mode = st.selectbox(
        "Choisir le mode",
        ["Pricing", "Implied Volatility", "Backtesting"],
        help="Pricing: valorisation | IV: calibration | Backtesting: simulation stratégie"
    )
    
    if mode == "Pricing":
        st.markdown("---")
        st.markdown("### Méthode de pricing")
        
        pricing_method = st.selectbox(
            "Modèle",
            ["Black-Scholes", "Monte Carlo"],
            help="Choisir la méthode de valorisation"
        )
    
    st.markdown("---")
    st.markdown("### Paramètres")

    S     = st.number_input("Spot S ($)",              value=100.0, step=1.0)
    K     = st.number_input("Strike K ($)",            value=100.0, step=1.0)
    T_day = st.number_input("Maturité (jours)",        value=30,    step=1, min_value=1)
    r     = st.number_input("Taux sans risque r (%)",  value=5.0,   step=0.1) / 100
    
    if mode != "Implied Volatility":
        sigma = st.number_input("Volatilité σ (%)",    value=20.0,  step=0.5) / 100
    
    q     = st.number_input("Dividend yield q (%)",    value=0.0,   step=0.1) / 100
    
    if mode == "Pricing":
        prem  = st.number_input("Prime payée ($) [opt.]",  value=0.0,   step=0.01)
    
    opt   = st.radio("Type d'option", ["call", "put"], horizontal=True)
    
    # Paramètres spécifiques à chaque mode
    if mode == "Pricing" and pricing_method == "Monte Carlo":
        st.markdown("---")
        st.markdown("### Paramètres Monte Carlo")
        n_sims = st.selectbox("Nombre de simulations", [10000, 50000, 100000, 250000, 500000], index=2)
        n_steps = st.selectbox("Nombre de pas de temps", [50, 100, 252, 500], index=2)
        antithetic = st.checkbox("Variables antithétiques", value=True)
        seed = st.number_input("Seed aléatoire", value=42, step=1)
    
    elif mode == "Implied Volatility":
        st.markdown("---")
        st.markdown("### Prix de marché")
        market_price = st.number_input("Prix observé ($)", value=5.0, step=0.01, min_value=0.01)
    
    elif mode == "Backtesting":
        st.markdown("---")
        st.markdown("### Paramètres Backtest")
        sigma = st.number_input("Volatilité σ (%)",    value=20.0,  step=0.5) / 100
        strategy = st.selectbox(
            "Stratégie",
            ["long_call", "long_put", "covered_call", "protective_put", "straddle", "strangle"],
            format_func=lambda x: {
                "long_call": "Long Call",
                "long_put": "Long Put",
                "covered_call": "Covered Call",
                "protective_put": "Protective Put",
                "straddle": "Straddle",
                "strangle": "Strangle"
            }[x]
        )
        initial_capital = st.number_input("Capital initial ($)", value=10000, step=100)
        backtest_days = st.slider("Horizon (jours)", 1, 365, T_day)
        n_simulations = st.selectbox("Nombre de simulations", [100, 500, 1000, 5000], index=2)

    st.markdown("---")
    run = st.button("⚡ RUN", use_container_width=True, type="primary")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if mode == "Pricing":
    st.markdown(f"# ◈ Options Pricer — {pricing_method}")
elif mode == "Implied Volatility":
    st.markdown("# ◈ Implied Volatility Calibrator")
else:
    st.markdown("# ◈ Strategy Backtester")

st.markdown(
    '<div class="author-link">by <a href="https://www.linkedin.com/in/arthurcotten/" target="_blank">Arthur Cotten</a> • '
    '<a href="https://github.com/arthurcotten" target="_blank">@arthurcotten</a></div>',
    unsafe_allow_html=True
)
st.markdown("---")

# ═══ MODE: PRICING ════════════════════════════════════════════════════════════

if mode == "Pricing":
    if run or True:
        T = T_day / 365
        
        if pricing_method == "Black-Scholes":
            price  = bs(S, K, T, r, sigma, q, opt)
            g      = greeks(S, K, T, r, sigma, q, opt)
            std_error = None
            mc_paths = None
        else:
            with st.spinner('Calcul Monte Carlo en cours...'):
                mc_result = monte_carlo_pricer(S, K, T, r, sigma, q, opt, n_sims, n_steps, antithetic, seed)
                price = mc_result["price"]
                std_error = mc_result["std_error"]
                g = {k: mc_result[k] for k in ["delta","gamma","vega","theta","rho"]}
                mc_paths = mc_result["paths"]
        
        prob   = prob_itm(S, K, T, r, sigma, q, opt)
        cost   = prem if prem > 0 else price
        be     = (K + cost) if opt=="call" else (K - cost)
        intrin = max(S-K, 0) if opt=="call" else max(K-S, 0)
        tv     = price - intrin
        moneyness = S/K
        mon_lbl = ("ATM" if abs(moneyness-1)<0.01
                   else "ITM" if (opt=="call" and moneyness>1) or (opt=="put" and moneyness<1)
                   else "OTM")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Prix", f"${price:.4f}")
        if std_error is not None:
            c2.metric("Std Error (MC)", f"${std_error:.4f}")
        else:
            c2.metric("Break-even", f"${be:.2f}")
        c3.metric("Prob ITM", f"{prob*100:.1f}%")
        c4.metric("Time Value", f"${tv:.4f}")
        c5.metric("Moneyness", mon_lbl)
        c6.metric("Intrinsèque", f"${intrin:.4f}")

        st.markdown("---")
        st.markdown("### Greeks — 1er ordre")
        gc1, gc2, gc3, gc4, gc5 = st.columns(5)
        gc1.metric("Delta Δ",  f"{g['delta']:+.5f}", help="Sensibilité au spot")
        gc2.metric("Gamma Γ",  f"{g['gamma']:.6f}",  help="Convexité du delta")
        gc3.metric("Vega ν",   f"{g['vega']:.5f}",   help="Sensibilité à la vol")
        gc4.metric("Theta Θ",  f"{g['theta']:+.5f}", help="Perte de valeur par jour")
        gc5.metric("Rho ρ",    f"{g['rho']:+.5f}",   help="Sensibilité au taux")

        st.markdown("---")
        alerts = []
        if T < 7/365:              alerts.append("⚠️ Maturité < 7 jours : theta élevé")
        if abs(g["delta"]) < 0.10: alerts.append("⚠️ Delta faible : option très OTM")
        if tv < 0.005:             alerts.append("⚠️ Time value quasi-nulle")
        if prob < 0.15:            alerts.append("⚠️ Probabilité ITM < 15%")
        if alerts:
            for a in alerts:
                st.warning(a)
        else:
            st.success("✓ Paramètres cohérents")

        S_range = np.linspace(S*0.68, S*1.32, 350)
        
        if pricing_method == "Black-Scholes":
            col1, col2, col3 = st.columns([2, 1, 1])
        else:
            col1, col2, col3 = st.columns([1.5, 1, 1])

        with col1:
            fig1, ax = plt.subplots(figsize=(7, 3.5), facecolor=BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): 
                sp.set_edgecolor(BORDER)
                sp.set_linewidth(1.5)
            pnl = (np.maximum(S_range-K, 0) - cost if opt=="call"
                   else np.maximum(K-S_range, 0) - cost)
            ax.axhline(0, color=GRAY, lw=1.5, alpha=0.7)
            ax.axvline(S,  color=GRAY,   lw=1.2, linestyle=":",  alpha=0.8)
            ax.axvline(K,  color=YELLOW, lw=2, linestyle="--", alpha=0.95, label=f"Strike ${K:.0f}")
            ax.axvline(be, color=GREEN,  lw=2.2, linestyle="--", alpha=0.95, label=f"BE ${be:.2f}")
            ax.fill_between(S_range, pnl, 0, where=pnl>=0, alpha=0.3, color=GREEN)
            ax.fill_between(S_range, pnl, 0, where=pnl<0,  alpha=0.3, color=RED)
            ax.plot(S_range, pnl, color=ACCENT, lw=2.5, label="P&L expiration")
            if pricing_method == "Black-Scholes":
                ax.plot(S_range, [bs(s,K,T,r,sigma,q,opt)-cost for s in S_range],
                        color=PURPLE, lw=2, linestyle="--", alpha=0.9, label="Valeur actuelle")
            else:
                ax.plot(S_range, [bs(s,K,T,r,sigma,q,opt)-cost for s in S_range],
                        color=PURPLE, lw=2, linestyle=":", alpha=0.7, label="Valeur actuelle (BS)")
            ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, framealpha=0.95)
            sty(ax, f"P&L à expiration · {opt.upper()} · Prime ${cost:.4f}", "Spot ($)", "P&L ($)")
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

        with col2:
            if pricing_method == "Monte Carlo" and mc_paths is not None:
                fig2, ax = plt.subplots(figsize=(3.5, 3.5), facecolor=BG)
                ax.set_facecolor(PANEL)
                for sp in ax.spines.values(): 
                    sp.set_edgecolor(BORDER)
                    sp.set_linewidth(1.5)
                ax.hist(mc_paths, bins=50, color=CYAN, alpha=0.7, edgecolor=CYAN, linewidth=0.5)
                ax.axvline(K, color=YELLOW, lw=2, linestyle="--", alpha=0.9, label=f"Strike ${K:.0f}")
                ax.axvline(np.mean(mc_paths), color=ACCENT, lw=2, linestyle="-", alpha=0.9, label=f"Moyenne ${np.mean(mc_paths):.2f}")
                ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, framealpha=0.95)
                sty(ax, f"Distribution S(T) · {n_sims:,} sims", "Prix terminal ($)", "Fréquence")
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)
            else:
                fig2, ax = plt.subplots(figsize=(3.5, 3.5), facecolor=BG)
                ax.set_facecolor(PANEL)
                for sp in ax.spines.values(): 
                    sp.set_edgecolor(BORDER)
                    sp.set_linewidth(1.5)
                vol_r  = np.linspace(0.01, sigma*3.5, 250)
                pr_vol = [bs(S,K,T,r,v,q,opt) for v in vol_r]
                ax.plot(vol_r*100, pr_vol, color=YELLOW, lw=2.5)
                ax.axvline(sigma*100, color=ACCENT, lw=2, linestyle="--", alpha=0.9)
                ax.axhline(price, color=ACCENT, lw=1.2, linestyle="--", alpha=0.7)
                ax.scatter([sigma*100], [price], color=ACCENT, s=80, zorder=5, edgecolors='white', linewidths=1.5)
                ax.fill_between(vol_r*100, pr_vol, alpha=0.2, color=YELLOW)
                sty(ax, f"Prix vs Vol [${price:.4f}]", "Vol (%)", "Prix ($)")
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

        with col3:
            fig3, ax = plt.subplots(figsize=(3.5, 3.5), facecolor=BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): 
                sp.set_edgecolor(BORDER)
                sp.set_linewidth(1.5)
            T_range = np.linspace(max(T, 1/365), max(T*5, 90/365), 200)
            pr_time = [bs(S,K,t,r,sigma,q,opt) for t in T_range]
            ax.plot(T_range*365, pr_time, color=RED, lw=2.5)
            ax.axvline(T*365, color=ACCENT, lw=2, linestyle="--", alpha=0.9)
            ax.scatter([T*365], [price], color=ACCENT, s=80, zorder=5, edgecolors='white', linewidths=1.5)
            ax.fill_between(T_range*365, pr_time, alpha=0.2, color=RED)
            sty(ax, f"Prix vs Temps [T={T_day}j]", "Jours restants", "Prix ($)")
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

        st.markdown("### Sensibilités Greeks vs Spot")
        col4, col5, col6, col7 = st.columns(4)
        greek_cfg = [
            (col4, "delta", "Delta Δ",  GREEN,  f"[{g['delta']:+.4f}]"),
            (col5, "gamma", "Gamma Γ",  YELLOW, f"[{g['gamma']:.5f}]"),
            (col6, "vega",  "Vega ν",   PURPLE, f"[{g['vega']:.4f}]"),
            (col7, "theta", "Theta Θ",  RED,    f"[{g['theta']:+.4f}]"),
        ]
        for col, gname, gtitle, gcol, gval in greek_cfg:
            with col:
                fig_g, ax = plt.subplots(figsize=(3.5, 3), facecolor=BG)
                ax.set_facecolor(PANEL)
                for sp in ax.spines.values(): 
                    sp.set_edgecolor(BORDER)
                    sp.set_linewidth(1.5)
                vals = [greeks(s,K,T,r,sigma,q,opt)[gname] for s in S_range]
                ax.plot(S_range, vals, color=gcol, lw=2.5)
                ax.axvline(S, color=GRAY, lw=1.2, linestyle=":", alpha=0.8)
                ax.axhline(g[gname], color=gcol, lw=1.2, linestyle="--", alpha=0.7)
                ax.fill_between(S_range, vals, alpha=0.2, color=gcol)
                sty(ax, f"{gtitle} {gval}", "Spot ($)", gname.capitalize())
                st.pyplot(fig_g, use_container_width=True)
                plt.close(fig_g)

# ═══ MODE: IMPLIED VOLATILITY ════════════════════════════════════════════════

elif mode == "Implied Volatility":
    if run or True:
        T = T_day / 365
        
        with st.spinner('Calibration de la volatilité implicite...'):
            iv = implied_volatility(market_price, S, K, T, r, q, opt)
        
        if np.isnan(iv):
            st.error("❌ Impossible de calibrer la volatilité implicite. Vérifiez que le prix de marché est cohérent.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Vol Implicite", f"{iv*100:.2f}%")
            col2.metric("Prix de marché", f"${market_price:.4f}")
            col3.metric("Prix théorique (IV)", f"${bs(S, K, T, r, iv, q, opt):.4f}")
            
            # Vega pour montrer la sensibilité
            g_iv = greeks(S, K, T, r, iv, q, opt)
            col4.metric("Vega", f"{g_iv['vega']:.5f}")
            
            st.markdown("---")
            
            # Smile de volatilité
            st.markdown("### Volatility Smile")
            strikes = np.linspace(S*0.7, S*1.3, 15)
            ivs = []
            for strike in strikes:
                theoretical_price = bs(S, strike, T, r, iv, q, opt)
                ivs.append(implied_volatility(theoretical_price, S, strike, T, r, q, opt))
            
            col_smile, col_surface = st.columns(2)
            
            with col_smile:
                fig_smile, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
                ax.set_facecolor(PANEL)
                for sp in ax.spines.values(): 
                    sp.set_edgecolor(BORDER)
                    sp.set_linewidth(1.5)
                ax.plot(strikes/S, np.array(ivs)*100, color=PURPLE, lw=2.5, marker='o', markersize=4)
                ax.axvline(1.0, color=GRAY, lw=1.2, linestyle=":", alpha=0.7, label="ATM")
                ax.axhline(iv*100, color=ACCENT, lw=1.2, linestyle="--", alpha=0.7, label=f"IV calibrée: {iv*100:.2f}%")
                ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
                sty(ax, "Volatility Smile", "Moneyness (K/S)", "Vol Implicite (%)")
                st.pyplot(fig_smile, use_container_width=True)
                plt.close(fig_smile)
            
            with col_surface:
                # Term structure
                maturities = np.linspace(max(T, 7/365), T*3, 10)
                term_ivs = []
                for mat in maturities:
                    theoretical_price = bs(S, K, mat, r, iv, q, opt)
                    term_ivs.append(implied_volatility(theoretical_price, S, K, mat, r, q, opt))
                
                fig_term, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
                ax.set_facecolor(PANEL)
                for sp in ax.spines.values(): 
                    sp.set_edgecolor(BORDER)
                    sp.set_linewidth(1.5)
                ax.plot(maturities*365, np.array(term_ivs)*100, color=CYAN, lw=2.5, marker='s', markersize=4)
                ax.axvline(T*365, color=GRAY, lw=1.2, linestyle=":", alpha=0.7, label=f"Maturité actuelle: {T_day}j")
                ax.axhline(iv*100, color=ACCENT, lw=1.2, linestyle="--", alpha=0.7)
                ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
                sty(ax, "Term Structure", "Maturité (jours)", "Vol Implicite (%)")
                st.pyplot(fig_term, use_container_width=True)
                plt.close(fig_term)

# ═══ MODE: BACKTESTING ═══════════════════════════════════════════════════════

elif mode == "Backtesting":
    if run or True:
        T = T_day / 365
        
        with st.spinner(f'Backtesting {strategy} sur {n_simulations} simulations...'):
            results_df = backtest_strategy(strategy, S, K, T, r, sigma, q, initial_capital, backtest_days, n_simulations)
        
        # Statistiques globales
        mean_pnl = results_df['pnl'].mean()
        median_pnl = results_df['pnl'].median()
        std_pnl = results_df['pnl'].std()
        win_rate = (results_df['pnl'] > 0).sum() / len(results_df) * 100
        max_gain = results_df['pnl'].max()
        max_loss = results_df['pnl'].min()
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
        
        st.markdown("### Statistiques de la stratégie")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("P&L Moyen", f"${mean_pnl:.2f}", delta=f"{mean_pnl/initial_capital*100:.1f}%")
        c2.metric("P&L Médian", f"${median_pnl:.2f}")
        c3.metric("Win Rate", f"{win_rate:.1f}%")
        c4.metric("Max Gain", f"${max_gain:.2f}")
        c5.metric("Max Loss", f"${max_loss:.2f}")
        c6.metric("Sharpe Ratio", f"{sharpe:.3f}")
        
        st.markdown("---")
        
        # Graphiques
        col_hist, col_cum = st.columns(2)
        
        with col_hist:
            fig_hist, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): 
                sp.set_edgecolor(BORDER)
                sp.set_linewidth(1.5)
            ax.hist(results_df['pnl'], bins=50, color=CYAN, alpha=0.7, edgecolor=CYAN, linewidth=0.5)
            ax.axvline(mean_pnl, color=ACCENT, lw=2, linestyle="--", alpha=0.9, label=f"Moyenne: ${mean_pnl:.2f}")
            ax.axvline(0, color=GRAY, lw=1.5, linestyle="-", alpha=0.7, label="Break-even")
            ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
            sty(ax, f"Distribution P&L · {strategy.replace('_', ' ').title()}", "P&L ($)", "Fréquence")
            st.pyplot(fig_hist, use_container_width=True)
            plt.close(fig_hist)
        
        with col_cum:
            fig_spot, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): 
                sp.set_edgecolor(BORDER)
                sp.set_linewidth(1.5)
            ax.scatter(results_df['final_spot'], results_df['pnl'], alpha=0.5, s=20, color=PURPLE)
            ax.axhline(0, color=GRAY, lw=1.5, linestyle="-", alpha=0.7, label="Break-even")
            ax.axvline(S, color=YELLOW, lw=1.2, linestyle="--", alpha=0.8, label=f"Spot initial: ${S:.2f}")
            ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
            sty(ax, "P&L vs Spot Final", "Spot final ($)", "P&L ($)")
            st.pyplot(fig_spot, use_container_width=True)
            plt.close(fig_spot)
        
        st.markdown("---")
        
        # Tableau des percentiles
        st.markdown("### Distribution des résultats")
        percentiles = [5, 25, 50, 75, 95]
        pct_data = {
            "Percentile": [f"{p}%" for p in percentiles],
            "P&L ($)": [f"${results_df['pnl'].quantile(p/100):.2f}" for p in percentiles],
            "Return (%)": [f"{results_df['return_pct'].quantile(p/100):.2f}%" for p in percentiles]
        }
        st.dataframe(pd.DataFrame(pct_data), use_container_width=True, hide_index=True)

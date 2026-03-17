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
    """Calibration de la volatilité implicite par méthode de Brent"""
    if T <= 1e-10:
        return np.nan
    
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

# ─── MONTE CARLO OPTIMISÉ ─────────────────────────────────────────────────────

@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def monte_carlo_pricer_cached(S, K, T, r, sigma, q, opt, n_sims, n_steps, antithetic, seed):
    """Version cachée du pricer Monte Carlo pour éviter recalculs"""
    return monte_carlo_pricer(S, K, T, r, sigma, q, opt, n_sims, n_steps, antithetic, seed)

def monte_carlo_pricer(S, K, T, r, sigma, q=0.0, opt="call", n_sims=100000, n_steps=252, antithetic=True, seed=42):
    """Monte Carlo optimisé avec gestion mémoire"""
    np.random.seed(seed)
    dt = T / n_steps
    
    # Limiter les paths stockés pour économiser la mémoire
    max_paths_to_store = min(1000, n_sims)
    
    # Calcul par batch si trop de simulations (évite saturation mémoire)
    batch_size = min(n_sims, 100000)
    n_batches = int(np.ceil(n_sims / batch_size))
    
    all_payoffs = []
    sample_paths = None
    
    for batch in range(n_batches):
        n_paths_batch = min(batch_size, n_sims - batch * batch_size)
        n_paths = n_paths_batch // 2 if antithetic else n_paths_batch
        
        Z = np.random.standard_normal((n_paths, n_steps))
        if antithetic:
            Z = np.concatenate([Z, -Z], axis=0)
        
        drift = (r - q - 0.5*sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        log_returns = drift + diffusion * Z
        log_price_paths = np.log(S) + np.cumsum(log_returns, axis=1)
        S_T = np.exp(log_price_paths[:, -1])
        
        # Conserver quelques paths du premier batch seulement
        if batch == 0:
            sample_paths = S_T[:max_paths_to_store].copy()
        
        if opt == "call":
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        all_payoffs.append(payoffs)
        
        # Libérer mémoire
        del Z, log_returns, log_price_paths, S_T, payoffs
    
    all_payoffs = np.concatenate(all_payoffs)
    price = np.exp(-r*T) * np.mean(all_payoffs)
    std_error = np.exp(-r*T) * np.std(all_payoffs) / np.sqrt(len(all_payoffs))
    
    # Greeks simplifiés (pas de recalcul complet pour économiser du temps)
    # On utilise des approximations analytiques quand possible
    g_bs = greeks(S, K, T, r, sigma, q, opt)
    
    return {
        "price": price,
        "std_error": std_error,
        "delta": g_bs["delta"],  # Utilise BS pour les Greeks (plus rapide)
        "gamma": g_bs["gamma"],
        "vega": g_bs["vega"],
        "theta": g_bs["theta"],
        "rho": g_bs["rho"],
        "paths": sample_paths
    }

# ─── BACKTESTING OPTIMISÉ ─────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def backtest_strategy_cached(strategy, S0, K, T, r, sigma, q, initial_capital, n_days, n_sims):
    """Version cachée du backtester"""
    return backtest_strategy(strategy, S0, K, T, r, sigma, q, initial_capital, n_days, n_sims)

def backtest_strategy(strategy, S0, K, T, r, sigma, q, initial_capital, n_days, n_sims=1000):
    """Backtest vectorisé pour performance"""
    np.random.seed(42)
    dt = 1/252
    
    # Simulation vectorisée du sous-jacent pour toutes les trajectoires à la fois
    Z = np.random.standard_normal((n_sims, n_days))
    drift = (r - q - 0.5*sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    log_returns = drift + diffusion * Z
    log_price_paths = np.log(S0) + np.cumsum(log_returns, axis=1)
    S_final = np.exp(log_price_paths[:, -1])
    
    # Calcul P&L selon stratégie
    results = []
    
    for i, S in enumerate(S_final):
        capital = initial_capital
        time_to_expiry = max(T - n_days/252, 0)
        
        if strategy == "long_call":
            entry_price = bs(S0, K, T, r, sigma, q, "call")
            payoff = max(S - K, 0)
            pnl = capital - entry_price + payoff
            
        elif strategy == "long_put":
            entry_price = bs(S0, K, T, r, sigma, q, "put")
            payoff = max(K - S, 0)
            pnl = capital - entry_price + payoff
            
        elif strategy == "covered_call":
            call_price = bs(S0, K, T, r, sigma, q, "call")
            call_payoff = -max(S - K, 0)
            pnl = capital - S0 + call_price + S + call_payoff
            
        elif strategy == "protective_put":
            put_price = bs(S0, K, T, r, sigma, q, "put")
            put_payoff = max(K - S, 0)
            pnl = capital - S0 - put_price + S + put_payoff
            
        elif strategy == "straddle":
            call_price = bs(S0, K, T, r, sigma, q, "call")
            put_price = bs(S0, K, T, r, sigma, q, "put")
            call_payoff = max(S - K, 0)
            put_payoff = max(K - S, 0)
            pnl = capital - call_price - put_price + call_payoff + put_payoff
            
        elif strategy == "strangle":
            K_call = K * 1.05
            K_put = K * 0.95
            call_price = bs(S0, K_call, T, r, sigma, q, "call")
            put_price = bs(S0, K_put, T, r, sigma, q, "put")
            call_payoff = max(S - K_call, 0)
            put_payoff = max(K_put - S, 0)
            pnl = capital - call_price - put_price + call_payoff + put_payoff
        
        results.append({
            "final_spot": S,
            "pnl": pnl - initial_capital,
            "return_pct": (pnl - initial_capital) / initial_capital * 100
        })
    
    return pd.DataFrame(results)

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
    
    # Paramètres spécifiques
    if mode == "Pricing" and pricing_method == "Monte Carlo":
        st.markdown("---")
        st.markdown("### Paramètres Monte Carlo")
        n_sims = st.selectbox(
            "Nombre de simulations",
            [10000, 50000, 100000, 250000],  # Limité à 250k pour éviter timeout
            index=2,
            help="⚠️ >250k peut causer des timeouts"
        )
        n_steps = st.selectbox("Nombre de pas", [50, 100, 252], index=2)
        antithetic = st.checkbox("Variables antithétiques", value=True)
        seed = st.number_input("Seed", value=42, step=1)
    
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
            format_func=lambda x: x.replace('_', ' ').title()
        )
        initial_capital = st.number_input("Capital initial ($)", value=10000, step=100)
        backtest_days = st.slider("Horizon (jours)", 1, 365, T_day)
        n_simulations = st.selectbox("Simulations", [100, 500, 1000, 2000], index=2)

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
            with st.spinner('Calcul Monte Carlo...'):
                try:
                    mc_result = monte_carlo_pricer_cached(S, K, T, r, sigma, q, opt, n_sims, n_steps, antithetic, seed)
                    price = mc_result["price"]
                    std_error = mc_result["std_error"]
                    g = {k: mc_result[k] for k in ["delta","gamma","vega","theta","rho"]}
                    mc_paths = mc_result["paths"]
                except Exception as e:
                    st.error(f"❌ Erreur Monte Carlo: {str(e)}")
                    st.info("💡 Essayez de réduire le nombre de simulations")
                    st.stop()
        
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
        gc1.metric("Delta Δ",  f"{g['delta']:+.5f}")
        gc2.metric("Gamma Γ",  f"{g['gamma']:.6f}")
        gc3.metric("Vega ν",   f"{g['vega']:.5f}")
        gc4.metric("Theta Θ",  f"{g['theta']:+.5f}")
        gc5.metric("Rho ρ",    f"{g['rho']:+.5f}")

        st.markdown("---")
        alerts = []
        if T < 7/365:              alerts.append("⚠️ Maturité < 7 jours")
        if abs(g["delta"]) < 0.10: alerts.append("⚠️ Delta très faible")
        if tv < 0.005:             alerts.append("⚠️ Time value nulle")
        if prob < 0.15:            alerts.append("⚠️ Prob ITM < 15%")
        
        for a in alerts:
            st.warning(a)
        if not alerts:
            st.success("✓ Paramètres OK")

        # Graphiques simplifiés pour performance
        S_range = np.linspace(S*0.7, S*1.3, 200)  # Réduit de 350 à 200 points
        
        col1, col2 = st.columns([2, 1])

        with col1:
            fig1, ax = plt.subplots(figsize=(7, 3.5), facecolor=BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): 
                sp.set_edgecolor(BORDER)
                sp.set_linewidth(1.5)
            pnl = (np.maximum(S_range-K, 0) - cost if opt=="call"
                   else np.maximum(K-S_range, 0) - cost)
            ax.axhline(0, color=GRAY, lw=1.5, alpha=0.7)
            ax.axvline(K,  color=YELLOW, lw=2, linestyle="--", alpha=0.95, label=f"Strike ${K:.0f}")
            ax.axvline(be, color=GREEN,  lw=2.2, linestyle="--", alpha=0.95, label=f"BE ${be:.2f}")
            ax.fill_between(S_range, pnl, 0, where=pnl>=0, alpha=0.3, color=GREEN)
            ax.fill_between(S_range, pnl, 0, where=pnl<0,  alpha=0.3, color=RED)
            ax.plot(S_range, pnl, color=ACCENT, lw=2.5, label="P&L expiration")
            ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
            sty(ax, f"P&L · {opt.upper()}", "Spot ($)", "P&L ($)")
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

        with col2:
            if pricing_method == "Monte Carlo" and mc_paths is not None:
                fig2, ax = plt.subplots(figsize=(3.5, 3.5), facecolor=BG)
                ax.set_facecolor(PANEL)
                for sp in ax.spines.values(): 
                    sp.set_edgecolor(BORDER)
                    sp.set_linewidth(1.5)
                ax.hist(mc_paths, bins=40, color=CYAN, alpha=0.7, edgecolor=CYAN, linewidth=0.5)
                ax.axvline(K, color=YELLOW, lw=2, linestyle="--", alpha=0.9)
                sty(ax, f"Distribution S(T)", "Prix terminal ($)", "Freq")
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

# ═══ MODE: IMPLIED VOLATILITY ════════════════════════════════════════════════

elif mode == "Implied Volatility":
    if run or True:
        T = T_day / 365
        
        with st.spinner('Calibration IV...'):
            iv = implied_volatility(market_price, S, K, T, r, q, opt)
        
        if np.isnan(iv):
            st.error("❌ Impossible de calibrer l'IV")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Vol Implicite", f"{iv*100:.2f}%")
            col2.metric("Prix marché", f"${market_price:.4f}")
            col3.metric("Prix théorique", f"${bs(S, K, T, r, iv, q, opt):.4f}")
            
            g_iv = greeks(S, K, T, r, iv, q, opt)
            col4.metric("Vega", f"{g_iv['vega']:.5f}")
            
            st.markdown("---")
            st.markdown("### Volatility Smile")
            
            strikes = np.linspace(S*0.75, S*1.25, 12)
            ivs = []
            for strike in strikes:
                theo_price = bs(S, strike, T, r, iv, q, opt)
                ivs.append(implied_volatility(theo_price, S, strike, T, r, q, opt))
            
            fig_smile, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): 
                sp.set_edgecolor(BORDER)
                sp.set_linewidth(1.5)
            ax.plot(strikes/S, np.array(ivs)*100, color=PURPLE, lw=2.5, marker='o', markersize=4)
            ax.axvline(1.0, color=GRAY, lw=1.2, linestyle=":", alpha=0.7, label="ATM")
            ax.axhline(iv*100, color=ACCENT, lw=1.2, linestyle="--", alpha=0.7, label=f"IV: {iv*100:.2f}%")
            ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
            sty(ax, "Volatility Smile", "Moneyness (K/S)", "IV (%)")
            st.pyplot(fig_smile, use_container_width=True)
            plt.close(fig_smile)

# ═══ MODE: BACKTESTING ═══════════════════════════════════════════════════════

elif mode == "Backtesting":
    if run or True:
        T = T_day / 365
        
        with st.spinner(f'Backtesting {n_simulations} simulations...'):
            try:
                results_df = backtest_strategy_cached(strategy, S, K, T, r, sigma, q, initial_capital, backtest_days, n_simulations)
            except Exception as e:
                st.error(f"❌ Erreur backtest: {str(e)}")
                st.stop()
        
        mean_pnl = results_df['pnl'].mean()
        median_pnl = results_df['pnl'].median()
        std_pnl = results_df['pnl'].std()
        win_rate = (results_df['pnl'] > 0).sum() / len(results_df) * 100
        max_gain = results_df['pnl'].max()
        max_loss = results_df['pnl'].min()
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
        
        st.markdown("### Statistiques")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("P&L Moyen", f"${mean_pnl:.2f}", delta=f"{mean_pnl/initial_capital*100:.1f}%")
        c2.metric("P&L Médian", f"${median_pnl:.2f}")
        c3.metric("Win Rate", f"{win_rate:.1f}%")
        c4.metric("Max Gain", f"${max_gain:.2f}")
        c5.metric("Max Loss", f"${max_loss:.2f}")
        c6.metric("Sharpe", f"{sharpe:.3f}")
        
        st.markdown("---")
        
        col_hist, col_scatter = st.columns(2)
        
        with col_hist:
            fig_hist, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): 
                sp.set_edgecolor(BORDER)
                sp.set_linewidth(1.5)
            ax.hist(results_df['pnl'], bins=40, color=CYAN, alpha=0.7)
            ax.axvline(mean_pnl, color=ACCENT, lw=2, linestyle="--", label=f"Moyenne")
            ax.axvline(0, color=GRAY, lw=1.5, linestyle="-", label="BE")
            ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
            sty(ax, f"Distribution P&L", "P&L ($)", "Fréquence")
            st.pyplot(fig_hist, use_container_width=True)
            plt.close(fig_hist)
        
        with col_scatter:
            fig_spot, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): 
                sp.set_edgecolor(BORDER)
                sp.set_linewidth(1.5)
            ax.scatter(results_df['final_spot'], results_df['pnl'], alpha=0.5, s=15, color=PURPLE)
            ax.axhline(0, color=GRAY, lw=1.5, linestyle="-")
            ax.axvline(S, color=YELLOW, lw=1.2, linestyle="--")
            sty(ax, "P&L vs Spot Final", "Spot final ($)", "P&L ($)")
            st.pyplot(fig_spot, use_container_width=True)
            plt.close(fig_spot)
        
        st.markdown("---")
        st.markdown("### Percentiles")
        percentiles = [5, 25, 50, 75, 95]
        pct_data = {
            "Percentile": [f"{p}%" for p in percentiles],
            "P&L ($)": [f"${results_df['pnl'].quantile(p/100):.2f}" for p in percentiles],
            "Return (%)": [f"{results_df['return_pct'].quantile(p/100):.2f}%" for p in percentiles]
        }
        st.dataframe(pd.DataFrame(pct_data), use_container_width=True, hide_index=True)

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
    /* === FOND GÉNÉRAL === */
    .main { 
        background-color: #000000 !important; 
    }
    .block-container { 
        padding-top: 1rem !important;
        background-color: #000000 !important;
    }
    [data-testid="stSidebar"] { 
        background-color: #0a0a0a !important; 
    }
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"] {
        background-color: #000000 !important;
    }
    
    /* === TITRES === */
    h1, h2, h3 { 
        color: #00ff00 !important; 
        font-family: monospace !important; 
        font-weight: bold !important;
        text-shadow: 0 0 10px rgba(0, 255, 0, 0.5);
    }
    
    /* === TEXTE GÉNÉRAL === */
    p, span, div, label, .stMarkdown { 
        color: #ffffff !important; 
    }
    
    /* === MÉTRIQUES === */
    div[data-testid="metric-container"] {
        background-color: #000000 !important;
        border: 3px solid #00ff00 !important;
        border-radius: 10px !important;
        padding: 16px !important;
        box-shadow: 0 0 15px rgba(0, 255, 0, 0.3) !important;
    }
    div[data-testid="metric-container"] label,
    div[data-testid="metric-container"] label p {
        color: #00ff00 !important;
        font-weight: bold !important;
        font-size: 1rem !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div,
    div[data-testid="stMetricValue"] p {
        color: #ffffff !important;
        font-weight: bold !important;
        font-size: 1.8rem !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    }
    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] svg {
        color: #00ff00 !important;
        fill: #00ff00 !important;
        font-weight: bold !important;
    }
    
    /* === ALERTES === */
    .stAlert,
    .stSuccess,
    .stWarning {
        background-color: #0a0a0a !important;
        border: 2px solid #00ff00 !important;
        color: #ffffff !important;
    }
    .stAlert p,
    .stSuccess p,
    .stWarning p {
        color: #ffffff !important;
    }
    
    /* === DATAFRAMES === */
    .dataframe {
        font-size: 1rem !important;
        font-family: monospace !important;
        background-color: #000000 !important;
        border: 2px solid #00ff00 !important;
    }
    .dataframe th {
        background-color: #0a0a0a !important;
        color: #00ff00 !important;
        font-weight: bold !important;
        border: 1px solid #00ff00 !important;
        padding: 8px !important;
    }
    .dataframe td {
        color: #ffffff !important;
        background-color: #000000 !important;
        border: 1px solid #333333 !important;
        padding: 8px !important;
    }
    
    /* === LIEN AUTEUR === */
    .author-link { 
        color: #888888 !important; 
        font-size: 0.9rem; 
        font-family: monospace; 
        margin-top: -10px;
        margin-bottom: 15px;
    }
    .author-link a {
        color: #00ff00 !important;
        text-decoration: none;
        font-weight: bold;
    }
    .author-link a:hover {
        color: #00ff00 !important;
        text-decoration: underline;
        text-shadow: 0 0 10px rgba(0, 255, 0, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# ─── STYLE MATPLOTLIB ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#000000",
    "axes.facecolor":   "#0a0a0a",
    "axes.edgecolor":   "#00ff00",
    "axes.labelcolor":  "#00ff00",
    "text.color":       "#ffffff",
    "xtick.color":      "#ffffff",
    "ytick.color":      "#ffffff",
    "grid.color":       "#333333",
    "grid.linewidth":   0.6,
    "font.family":      "monospace",
})

BG     = "#000000"
PANEL  = "#0a0a0a"
BORDER = "#00ff00"
ACCENT = "#00ff00"
GREEN  = "#00ff00"
RED    = "#ff0000"
YELLOW = "#ffff00"
PURPLE = "#ff00ff"
CYAN   = "#00ffff"
ORANGE = "#ff8800"
GRAY   = "#888888"
TEXT   = "#ffffff"
TITLE  = "#00ff00"

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
    if market_price < intrinsic * 0.99:  # Tolérance de 1%
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

@st.cache_data(ttl=300)
def monte_carlo_pricer_cached(S, K, T, r, sigma, q, opt, n_sims, n_steps, antithetic, seed):
    """Version cachée du pricer Monte Carlo"""
    return monte_carlo_pricer(S, K, T, r, sigma, q, opt, n_sims, n_steps, antithetic, seed)

def monte_carlo_pricer(S, K, T, r, sigma, q=0.0, opt="call", n_sims=100000, n_steps=252, antithetic=True, seed=42):
    """Monte Carlo optimisé avec gestion mémoire"""
    np.random.seed(seed)
    dt = T / n_steps
    
    max_paths_to_store = min(1000, n_sims)
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
        
        if batch == 0:
            sample_paths = S_T[:max_paths_to_store].copy()
        
        if opt == "call":
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        all_payoffs.append(payoffs)
        del Z, log_returns, log_price_paths, S_T, payoffs
    
    all_payoffs = np.concatenate(all_payoffs)
    price = np.exp(-r*T) * np.mean(all_payoffs)
    std_error = np.exp(-r*T) * np.std(all_payoffs) / np.sqrt(len(all_payoffs))
    
    g_bs = greeks(S, K, T, r, sigma, q, opt)
    
    return {
        "price": price,
        "std_error": std_error,
        "delta": g_bs["delta"],
        "gamma": g_bs["gamma"],
        "vega": g_bs["vega"],
        "theta": g_bs["theta"],
        "rho": g_bs["rho"],
        "paths": sample_paths
    }

# ─── BACKTESTING CORRIGÉ ──────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def backtest_strategy_cached(strategy, S0, K, T, r, sigma, q, initial_capital, n_days, n_sims):
    """Version cachée du backtester"""
    return backtest_strategy(strategy, S0, K, T, r, sigma, q, initial_capital, n_days, n_sims)

def backtest_strategy(strategy, S0, K, T, r, sigma, q, initial_capital, n_days, n_sims=1000):
    """
    Backtest corrigé : simulation complète du P&L jusqu'à expiration
    """
    np.random.seed(42)
    
    # Calcul du nombre de jours jusqu'à expiration
    days_to_expiry = int(T * 252)
    n_days = min(n_days, days_to_expiry)
    
    dt = 1/252
    time_remaining = T - (n_days / 252)
    
    # Simulation du sous-jacent
    Z = np.random.standard_normal((n_sims, n_days))
    drift = (r - q - 0.5*sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    log_returns = drift + diffusion * Z
    log_price_paths = np.log(S0) + np.cumsum(log_returns, axis=1)
    S_final = np.exp(log_price_paths[:, -1])
    
    results = []
    
    for i, S_end in enumerate(S_final):
        # Prix d'entrée au t=0
        if strategy == "long_call":
            entry_cost = bs(S0, K, T, r, sigma, q, "call")
            # Valeur à t=n_days
            if time_remaining > 0:
                exit_value = bs(S_end, K, time_remaining, r, sigma, q, "call")
            else:
                exit_value = max(S_end - K, 0)
            pnl = exit_value - entry_cost
            
        elif strategy == "long_put":
            entry_cost = bs(S0, K, T, r, sigma, q, "put")
            if time_remaining > 0:
                exit_value = bs(S_end, K, time_remaining, r, sigma, q, "put")
            else:
                exit_value = max(K - S_end, 0)
            pnl = exit_value - entry_cost
            
        elif strategy == "covered_call":
            # Achat stock + vente call
            stock_entry = S0
            call_premium = bs(S0, K, T, r, sigma, q, "call")
            
            stock_exit = S_end
            if time_remaining > 0:
                call_cost = bs(S_end, K, time_remaining, r, sigma, q, "call")
            else:
                call_cost = max(S_end - K, 0)
            
            pnl = (stock_exit - stock_entry) + call_premium - call_cost
            
        elif strategy == "protective_put":
            # Achat stock + achat put
            stock_entry = S0
            put_cost = bs(S0, K, T, r, sigma, q, "put")
            
            stock_exit = S_end
            if time_remaining > 0:
                put_value = bs(S_end, K, time_remaining, r, sigma, q, "put")
            else:
                put_value = max(K - S_end, 0)
            
            pnl = (stock_exit - stock_entry) + put_value - put_cost
            
        elif strategy == "straddle":
            # Achat call + achat put
            call_cost = bs(S0, K, T, r, sigma, q, "call")
            put_cost = bs(S0, K, T, r, sigma, q, "put")
            
            if time_remaining > 0:
                call_value = bs(S_end, K, time_remaining, r, sigma, q, "call")
                put_value = bs(S_end, K, time_remaining, r, sigma, q, "put")
            else:
                call_value = max(S_end - K, 0)
                put_value = max(K - S_end, 0)
            
            pnl = (call_value + put_value) - (call_cost + put_cost)
            
        elif strategy == "strangle":
            # Achat call OTM + achat put OTM
            K_call = K * 1.05
            K_put = K * 0.95
            
            call_cost = bs(S0, K_call, T, r, sigma, q, "call")
            put_cost = bs(S0, K_put, T, r, sigma, q, "put")
            
            if time_remaining > 0:
                call_value = bs(S_end, K_call, time_remaining, r, sigma, q, "call")
                put_value = bs(S_end, K_put, time_remaining, r, sigma, q, "put")
            else:
                call_value = max(S_end - K_call, 0)
                put_value = max(K_put - S_end, 0)
            
            pnl = (call_value + put_value) - (call_cost + put_cost)
        
        results.append({
            "final_spot": S_end,
            "pnl": pnl,
            "return_pct": (pnl / initial_capital) * 100 if initial_capital > 0 else 0
        })
    
    return pd.DataFrame(results)

# ─── HELPERS PLOT ─────────────────────────────────────────────────────────────

def sty(ax, title, xl, yl):
    ax.set_title(title, color=TITLE, fontsize=11, pad=10, fontweight="bold")
    ax.set_xlabel(xl, color=TITLE, fontsize=10, fontweight="bold")
    ax.set_ylabel(yl, color=TITLE, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.tick_params(labelsize=9, colors=TEXT, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor(BORDER)

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
            [10000, 50000, 100000, 250000],
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
        backtest_days = st.slider("Horizon (jours)", 1, min(365, T_day), min(T_day, 30))
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

        S_range = np.linspace(S*0.7, S*1.3, 200)
        
        col1, col2 = st.columns([2, 1])

        with col1:
            fig1, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): 
                sp.set_edgecolor(BORDER)
                sp.set_linewidth(2)
            pnl = (np.maximum(S_range-K, 0) - cost if opt=="call"
                   else np.maximum(K-S_range, 0) - cost)
            ax.axhline(0, color=GRAY, lw=2, alpha=0.7)
            ax.axvline(K,  color=YELLOW, lw=2.5, linestyle="--", alpha=0.95, label=f"Strike ${K:.0f}")
            ax.axvline(be, color=GREEN,  lw=2.5, linestyle="--", alpha=0.95, label=f"BE ${be:.2f}")
            ax.fill_between(S_range, pnl, 0, where=pnl>=0, alpha=0.3, color=GREEN)
            ax.fill_between(S_range, pnl, 0, where=pnl<0,  alpha=0.3, color=RED)
            ax.plot(S_range, pnl, color=ACCENT, lw=3, label="P&L expiration")
            ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
            sty(ax, f"P&L · {opt.upper()}", "Spot ($)", "P&L ($)")
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

        with col2:
            if pricing_method == "Monte Carlo" and mc_paths is not None:
                fig2, ax = plt.subplots(figsize=(4.5, 4.5), facecolor=BG)
                ax.set_facecolor(PANEL)
                for sp in ax.spines.values(): 
                    sp.set_edgecolor(BORDER)
                    sp.set_linewidth(2)
                ax.hist(mc_paths, bins=40, color=CYAN, alpha=0.7, edgecolor=CYAN, linewidth=0.5)
                ax.axvline(K, color=YELLOW, lw=2.5, linestyle="--", alpha=0.9)
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
            st.error("❌ Impossible de calibrer l'IV - vérifiez que le prix > valeur intrinsèque")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Vol Implicite", f"{iv*100:.2f}%")
            col2.metric("Prix marché", f"${market_price:.4f}")
            col3.metric("Prix théorique", f"${bs(S, K, T, r, iv, q, opt):.4f}")
            
            g_iv = greeks(S, K, T, r, iv, q, opt)
            col4.metric("Vega", f"{g_iv['vega']:.5f}")
            
            st.markdown("---")
            
            # Génération du smile complet (calls + puts)
            st.markdown("### Volatility Skew & Surface")
            
            col_skew, col_term = st.columns(2)
            
            with col_skew:
                # SKEW: Vol implicite vs Strike (Calls ET Puts)
                strikes = np.linspace(S*0.7, S*1.3, 20)
                iv_calls = []
                iv_puts = []
                
                for strike in strikes:
                    # Prix théoriques avec la vol calibrée
                    call_price = bs(S, strike, T, r, iv, q, "call")
                    put_price = bs(S, strike, T, r, iv, q, "put")
                    
                    # Recalibration de l'IV pour chaque strike
                    iv_call = implied_volatility(call_price, S, strike, T, r, q, "call")
                    iv_put = implied_volatility(put_price, S, strike, T, r, q, "put")
                    
                    iv_calls.append(iv_call if not np.isnan(iv_call) else None)
                    iv_puts.append(iv_put if not np.isnan(iv_put) else None)
                
                fig_skew, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
                ax.set_facecolor(PANEL)
                for sp in ax.spines.values(): 
                    sp.set_edgecolor(BORDER)
                    sp.set_linewidth(2)
                
                # Plot Calls
                valid_calls = [(k/S, v*100) for k, v in zip(strikes, iv_calls) if v is not None]
                if valid_calls:
                    x_calls, y_calls = zip(*valid_calls)
                    ax.plot(x_calls, y_calls, color=CYAN, lw=3, marker='o', markersize=6, label='Calls', alpha=0.9)
                
                # Plot Puts
                valid_puts = [(k/S, v*100) for k, v in zip(strikes, iv_puts) if v is not None]
                if valid_puts:
                    x_puts, y_puts = zip(*valid_puts)
                    ax.plot(x_puts, y_puts, color=PURPLE, lw=3, marker='s', markersize=6, label='Puts', alpha=0.9)
                
                ax.axvline(1.0, color=GRAY, lw=1.5, linestyle=":", alpha=0.7, label="ATM")
                ax.axhline(iv*100, color=ACCENT, lw=1.5, linestyle="--", alpha=0.7, label=f"IV ref: {iv*100:.2f}%")
                ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
                sty(ax, "Volatility Skew (Calls vs Puts)", "Moneyness (K/S)", "IV (%)")
                st.pyplot(fig_skew, use_container_width=True)
                plt.close(fig_skew)
            
            with col_term:
                # TERM STRUCTURE: Vol implicite vs Maturité
                maturities = np.linspace(max(T, 7/365), min(T*3, 1.0), 12)
                term_iv_calls = []
                term_iv_puts = []
                
                for mat in maturities:
                    call_price = bs(S, K, mat, r, iv, q, "call")
                    put_price = bs(S, K, mat, r, iv, q, "put")
                    
                    iv_call = implied_volatility(call_price, S, K, mat, r, q, "call")
                    iv_put = implied_volatility(put_price, S, K, mat, r, q, "put")
                    
                    term_iv_calls.append(iv_call if not np.isnan(iv_call) else None)
                    term_iv_puts.append(iv_put if not np.isnan(iv_put) else None)
                
                fig_term, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
                ax.set_facecolor(PANEL)
                for sp in ax.spines.values(): 
                    sp.set_edgecolor(BORDER)
                    sp.set_linewidth(2)
                
                valid_term_calls = [(m*365, v*100) for m, v in zip(maturities, term_iv_calls) if v is not None]
                if valid_term_calls:
                    x_term_c, y_term_c = zip(*valid_term_calls)
                    ax.plot(x_term_c, y_term_c, color=CYAN, lw=3, marker='o', markersize=6, label='Calls', alpha=0.9)
                
                valid_term_puts = [(m*365, v*100) for m, v in zip(maturities, term_iv_puts) if v is not None]
                if valid_term_puts:
                    x_term_p, y_term_p = zip(*valid_term_puts)
                    ax.plot(x_term_p, y_term_p, color=PURPLE, lw=3, marker='s', markersize=6, label='Puts', alpha=0.9)
                
                ax.axvline(T*365, color=GRAY, lw=1.5, linestyle=":", alpha=0.7, label=f"Maturité: {T_day}j")
                ax.axhline(iv*100, color=ACCENT, lw=1.5, linestyle="--", alpha=0.7)
                ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
                sty(ax, "Term Structure", "Maturité (jours)", "IV (%)")
                st.pyplot(fig_term, use_container_width=True)
                plt.close(fig_term)

# ═══ MODE: BACKTESTING ═══════════════════════════════════════════════════════

elif mode == "Backtesting":
    if run or True:
        T = T_day / 365
        
        with st.spinner(f'Backtesting {n_simulations} simulations...'):
            try:
                results_df = backtest_strategy_cached(strategy, S, K, T, r, sigma, q, initial_capital, backtest_days, n_simulations)
            except Exception as e:
                st.error(f"❌ Erreur backtest: {str(e)}")
                st.info(f"Debug: {e}")
                st.stop()
        
        mean_pnl = results_df['pnl'].mean()
        median_pnl = results_df['pnl'].median()
        std_pnl = results_df['pnl'].std()
        win_rate = (results_df['pnl'] > 0).sum() / len(results_df) * 100
        max_gain = results_df['pnl'].max()
        max_loss = results_df['pnl'].min()
        sharpe = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0
        
        st.markdown("### Statistiques de performance")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("P&L Moyen", f"${mean_pnl:.2f}", delta=f"{mean_pnl/abs(mean_pnl) if mean_pnl != 0 else 0:.1f}")
        c2.metric("P&L Médian", f"${median_pnl:.2f}")
        c3.metric("Win Rate", f"{win_rate:.1f}%")
        c4.metric("Max Gain", f"${max_gain:.2f}")
        c5.metric("Max Loss", f"${max_loss:.2f}")
        c6.metric("Sharpe", f"{sharpe:.3f}")
        
        st.markdown("---")
        
        col_hist, col_scatter = st.columns(2)
        
        with col_hist:
            fig_hist, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): 
                sp.set_edgecolor(BORDER)
                sp.set_linewidth(2)
            ax.hist(results_df['pnl'], bins=50, color=CYAN, alpha=0.7, edgecolor=CYAN, linewidth=0.5)
            ax.axvline(mean_pnl, color=ACCENT, lw=3, linestyle="--", label=f"Moyenne: ${mean_pnl:.0f}")
            ax.axvline(0, color=GRAY, lw=2.5, linestyle="-", alpha=0.8, label="Break-even")
            ax.legend(fontsize=10, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
            sty(ax, f"Distribution P&L · {strategy.replace('_', ' ').title()}", "P&L ($)", "Fréquence")
            st.pyplot(fig_hist, use_container_width=True)
            plt.close(fig_hist)
        
        with col_scatter:
            fig_spot, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): 
                sp.set_edgecolor(BORDER)
                sp.set_linewidth(2)
            ax.scatter(results_df['final_spot'], results_df['pnl'], alpha=0.6, s=30, color=PURPLE, edgecolors='white', linewidths=0.5)
            ax.axhline(0, color=GRAY, lw=2.5, linestyle="-", alpha=0.8, label="Break-even")
            ax.axvline(S, color=YELLOW, lw=2.5, linestyle="--", alpha=0.9, label=f"Spot initial: ${S:.0f}")
            ax.legend(fontsize=10, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
            sty(ax, "P&L vs Spot Final", "Spot final ($)", "P&L ($)")
            st.pyplot(fig_spot, use_container_width=True)
            plt.close(fig_spot)
        
        st.markdown("---")
        st.markdown("### Distribution des résultats")
        percentiles = [5, 25, 50, 75, 95]
        pct_data = {
            "Percentile": [f"{p}%" for p in percentiles],
            "P&L ($)": [f"${results_df['pnl'].quantile(p/100):.2f}" for p in percentiles],
            "Return (%)": [f"{results_df['return_pct'].quantile(p/100):.2f}%" for p in percentiles]
        }
        st.dataframe(pd.DataFrame(pct_data), use_container_width=True, hide_index=True)

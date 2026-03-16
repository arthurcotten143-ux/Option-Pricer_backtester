"""
Black-Scholes Pricer — Streamlit
Compatible GitHub Codespaces / navigateur
Lancer avec : streamlit run streamlit_bs_pricer.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BS Pricer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #080b12; }
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebar"] { background-color: #0f1219; }
    h1, h2, h3 { color: #38bdf8; font-family: monospace; }
    .metric-label { color: #5a6380 !important; font-size: 0.75rem !important; }
    .stMetric { background-color: #131720; border-radius: 8px; padding: 8px; }
    div[data-testid="metric-container"] {
        background-color: #131720;
        border: 1px solid #1e2235;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─── STYLE MATPLOTLIB ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1219",
    "axes.facecolor":   "#131720",
    "axes.edgecolor":   "#1e2235",
    "axes.labelcolor":  "#5a6380",
    "text.color":       "#e8ecf8",
    "xtick.color":      "#5a6380",
    "ytick.color":      "#5a6380",
    "grid.color":       "#1e2235",
    "grid.linewidth":   0.5,
    "font.family":      "monospace",
})

BG     = "#0f1219"
PANEL  = "#131720"
BORDER = "#1e2235"
ACCENT = "#38bdf8"
GREEN  = "#10b981"
RED    = "#f87171"
YELLOW = "#fbbf24"
PURPLE = "#a78bfa"
CYAN   = "#22d3ee"
ORANGE = "#fb923c"
GRAY   = "#5a6380"
TEXT   = "#e8ecf8"

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
        return {k: 0.0 for k in ["delta","gamma","vega","theta","rho","vanna","volga","charm"]}
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
    vanna = -np.exp(-q*T)*nd1*d2 / sigma
    volga = vega*d1*d2 / sigma
    charm = np.exp(-q*T)*nd1*(2*(r-q)*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
    return {"delta":delta, "gamma":gamma, "vega":vega, "theta":theta,
            "rho":rho, "vanna":vanna, "volga":volga, "charm":charm}

def prob_itm(S, K, T, r, sigma, q=0.0, opt="call"):
    if T <= 1e-10 or sigma <= 1e-10:
        return 1.0 if (opt=="call" and S>K) or (opt=="put" and S<K) else 0.0
    d2 = (np.log(S/K)+(r-q-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.cdf(d2) if opt=="call" else norm.cdf(-d2)

# ─── HELPERS PLOT ─────────────────────────────────────────────────────────────

def sty(ax, title, xl, yl):
    ax.set_title(title, color=TEXT, fontsize=8.5, pad=6, fontweight="bold")
    ax.set_xlabel(xl, color=GRAY, fontsize=7.5)
    ax.set_ylabel(yl, color=GRAY, fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ◈ BS PRICER")
    st.markdown("---")
    st.markdown("### Paramètres")

    S     = st.number_input("Spot S ($)",              value=100.0, step=1.0)
    K     = st.number_input("Strike K ($)",            value=100.0, step=1.0)
    T_day = st.number_input("Maturité (jours)",        value=30,    step=1, min_value=1)
    r     = st.number_input("Taux sans risque r (%)",  value=5.0,   step=0.1) / 100
    sigma = st.number_input("Volatilité σ (%)",        value=20.0,  step=0.5) / 100
    q     = st.number_input("Dividend yield q (%)",    value=0.0,   step=0.1) / 100
    prem  = st.number_input("Prime payée ($) [opt.]",  value=0.0,   step=0.01)
    opt   = st.radio("Type d'option", ["call", "put"], horizontal=True)

    st.markdown("---")
    run = st.button("⚡ PRICER", use_container_width=True, type="primary")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

st.markdown("# ◈ Black-Scholes Pricer")
st.markdown("---")

if run or True:  # calcul automatique au chargement
    T = T_day / 365
    price  = bs(S, K, T, r, sigma, q, opt)
    g      = greeks(S, K, T, r, sigma, q, opt)
    prob   = prob_itm(S, K, T, r, sigma, q, opt)
    cost   = prem if prem > 0 else price
    be     = (K + cost) if opt=="call" else (K - cost)
    intrin = max(S-K, 0) if opt=="call" else max(K-S, 0)
    tv     = price - intrin
    moneyness = S/K
    mon_lbl = ("ATM" if abs(moneyness-1)<0.01
               else "ITM" if (opt=="call" and moneyness>1) or (opt=="put" and moneyness<1)
               else "OTM")

    # ── Métriques principales ─────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Prix BS",       f"${price:.4f}")
    c2.metric("Break-even",    f"${be:.2f}")
    c3.metric("Prob ITM",      f"{prob*100:.1f}%")
    c4.metric("Time Value",    f"${tv:.4f}")
    c5.metric("Moneyness",     mon_lbl)
    c6.metric("Intrinsèque",   f"${intrin:.4f}")

    st.markdown("---")

    # ── Greeks 1er ordre ─────────────────────────────────────────────────────
    st.markdown("### Greeks — 1er ordre")
    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
    gc1.metric("Delta Δ",  f"{g['delta']:+.5f}", help="Sensibilité au spot")
    gc2.metric("Gamma Γ",  f"{g['gamma']:.6f}",  help="Convexité du delta")
    gc3.metric("Vega ν",   f"{g['vega']:.5f}",   help="Sensibilité à la vol ($/+1%σ)")
    gc4.metric("Theta Θ",  f"{g['theta']:+.5f}", help="Perte de valeur par jour")
    gc5.metric("Rho ρ",    f"{g['rho']:+.5f}",   help="Sensibilité au taux")

    # Greeks 2nd ordre
    st.markdown("### Greeks — 2ème ordre")
    gd1, gd2, gd3 = st.columns(3)
    gd1.metric("Vanna",  f"{g['vanna']:.6f}", help="δΔ/δσ — sensibilité croisée spot/vol")
    gd2.metric("Volga",  f"{g['volga']:.6f}", help="δVega/δσ — convexité vol")
    gd3.metric("Charm",  f"{g['charm']:.6f}", help="δΔ/δt — variation du delta dans le temps")

    st.markdown("---")

    # ── Alertes ───────────────────────────────────────────────────────────────
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

    # ── Graphiques ────────────────────────────────────────────────────────────
    S_range = np.linspace(S*0.68, S*1.32, 350)

    # Ligne 1 : P&L + Prix vs Vol + Prix vs Temps
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        fig1, ax = plt.subplots(figsize=(7, 3.5), facecolor=PANEL)
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        pnl = (np.maximum(S_range-K, 0) - cost if opt=="call"
               else np.maximum(K-S_range, 0) - cost)
        ax.axhline(0, color=BORDER, lw=1.2)
        ax.axvline(S,  color=GRAY,   lw=1,   linestyle=":",  alpha=0.7)
        ax.axvline(K,  color=YELLOW, lw=1.2, linestyle="--", alpha=0.9, label=f"Strike ${K:.0f}")
        ax.axvline(be, color=GREEN,  lw=1.5, linestyle="--", alpha=0.9, label=f"BE ${be:.2f}")
        ax.fill_between(S_range, pnl, 0, where=pnl>=0, alpha=0.22, color=GREEN)
        ax.fill_between(S_range, pnl, 0, where=pnl<0,  alpha=0.22, color=RED)
        ax.plot(S_range, pnl, color=ACCENT, lw=2.2, label="P&L expiration")
        ax.plot(S_range,
                [bs(s,K,T,r,sigma,q,opt)-cost for s in S_range],
                color=PURPLE, lw=1.4, linestyle="--", alpha=0.8, label="Valeur actuelle")
        ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
        sty(ax, f"P&L à expiration · {opt.upper()} · Prime ${cost:.4f}", "Spot ($)", "P&L ($)")
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with col2:
        fig2, ax = plt.subplots(figsize=(3.5, 3.5), facecolor=PANEL)
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        vol_r  = np.linspace(0.01, sigma*3.5, 250)
        pr_vol = [bs(S,K,T,r,v,q,opt) for v in vol_r]
        ax.plot(vol_r*100, pr_vol, color=YELLOW, lw=2)
        ax.axvline(sigma*100, color=ACCENT, lw=1.2, linestyle="--", alpha=0.8)
        ax.axhline(price,     color=ACCENT, lw=0.8, linestyle="--", alpha=0.5)
        ax.scatter([sigma*100], [price], color=ACCENT, s=50, zorder=5)
        ax.fill_between(vol_r*100, pr_vol, alpha=0.12, color=YELLOW)
        sty(ax, f"Prix vs Vol [${price:.4f}]", "Vol (%)", "Prix ($)")
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with col3:
        fig3, ax = plt.subplots(figsize=(3.5, 3.5), facecolor=PANEL)
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        T_range = np.linspace(max(T, 1/365), max(T*5, 90/365), 200)
        pr_time = [bs(S,K,t,r,sigma,q,opt) for t in T_range]
        ax.plot(T_range*365, pr_time, color=RED, lw=2)
        ax.axvline(T*365, color=ACCENT, lw=1.2, linestyle="--", alpha=0.8)
        ax.scatter([T*365], [price], color=ACCENT, s=50, zorder=5)
        ax.fill_between(T_range*365, pr_time, alpha=0.12, color=RED)
        sty(ax, f"Prix vs Temps [T={T_day}j]", "Jours restants", "Prix ($)")
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    # Ligne 2 : Greeks vs Spot
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
            fig_g, ax = plt.subplots(figsize=(3.5, 3), facecolor=PANEL)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            vals = [greeks(s,K,T,r,sigma,q,opt)[gname] for s in S_range]
            ax.plot(S_range, vals, color=gcol, lw=2)
            ax.axvline(S, color=GRAY, lw=1, linestyle=":", alpha=0.6)
            ax.axhline(g[gname], color=gcol, lw=0.8, linestyle="--", alpha=0.5)
            ax.fill_between(S_range, vals, alpha=0.13, color=gcol)
            sty(ax, f"{gtitle} {gval}", "Spot ($)", gname.capitalize())
            st.pyplot(fig_g, use_container_width=True)
            plt.close(fig_g)




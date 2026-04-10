"""
Options Pricer
streamlit run pricer.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st

st.set_page_config(page_title="Pricer", layout="centered")

st.markdown("""
<style>
.block-container{padding-top:3rem;padding-bottom:1rem;max-width:800px;}
div[data-testid="stVerticalBlock"]{gap:0.4rem;}
p{margin:0;font-size:14px;line-height:1.6;}
h1{font-size:2.2rem;margin-bottom:1.5rem;font-weight:600;}

/* Square inputs */
input, select, [data-baseweb="select"] > div {
    border-radius: 2px !important;
}
button {
    border-radius: 2px !important;
}
[data-testid="stNumberInput"] > div > div > input {
    border-radius: 2px !important;
}
[data-testid="stSelectbox"] > div > div {
    border-radius: 2px !important;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"#fafafa",
    "axes.facecolor":"#fafafa",
    "axes.edgecolor":"#ccc",
    "axes.linewidth":0.5,
    "font.size":8,
    "axes.labelsize":8,
    "xtick.labelsize":7,
    "ytick.labelsize":7,
    "grid.linewidth":0.3,
    "grid.alpha":0.5
})

def bs(S,K,T,r,sig,q=0,opt="call"):
    if T<=0:return max(S-K,0) if opt=="call" else max(K-S,0)
    d1=(np.log(S/K)+(r-q+0.5*sig**2)*T)/(sig*np.sqrt(T))
    d2=d1-sig*np.sqrt(T)
    if opt=="call":return S*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*np.exp(-q*T)*norm.cdf(-d1)

def greeks(S,K,T,r,sig,q=0,opt="call"):
    if T<=0 or sig<=0:return {"delta":0,"gamma":0,"vega":0,"theta":0,"rho":0}
    d1=(np.log(S/K)+(r-q+0.5*sig**2)*T)/(sig*np.sqrt(T))
    d2=d1-sig*np.sqrt(T)
    nd1=norm.pdf(d1)
    if opt=="call":
        delta=np.exp(-q*T)*norm.cdf(d1)
        theta=(-(S*np.exp(-q*T)*nd1*sig)/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))/365
        rho=K*T*np.exp(-r*T)*norm.cdf(d2)/100
    else:
        delta=-np.exp(-q*T)*norm.cdf(-d1)
        theta=(-(S*np.exp(-q*T)*nd1*sig)/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))/365
        rho=-K*T*np.exp(-r*T)*norm.cdf(-d2)/100
    gamma=np.exp(-q*T)*nd1/(S*sig*np.sqrt(T))
    vega=S*np.exp(-q*T)*nd1*np.sqrt(T)/100
    return {"delta":delta,"gamma":gamma,"vega":vega,"theta":theta,"rho":rho}

def mc(S,K,T,r,sig,q,opt,n,seed):
    np.random.seed(seed)
    Z=np.random.standard_normal(n)
    ST=S*np.exp((r-q-0.5*sig**2)*T+sig*np.sqrt(T)*Z)
    pay=np.maximum(ST-K,0) if opt=="call" else np.maximum(K-ST,0)
    return np.exp(-r*T)*np.mean(pay),np.exp(-r*T)*np.std(pay)/np.sqrt(n),ST[:300]

def skewed_iv(base_iv,S,K,T,skew_slope,skew_conv,term_slope):
    moneyness=1.0-K/S
    term_adj=T-0.25
    return max(base_iv+skew_slope*moneyness+skew_conv*(moneyness**2)+term_slope*term_adj,0.01)

def backtest(strat,S,K,T,r,sig,q,days,n):
    np.random.seed(42)
    dt=T/days
    Z=np.random.standard_normal((n,days))
    Sf=S*np.exp(np.sum((r-q-0.5*sig**2)*dt+sig*np.sqrt(dt)*Z,axis=1))
    pnls=[]
    for Se in Sf:
        if strat=="call":pnl=max(Se-K,0)-bs(S,K,T,r,sig,q,"call")
        elif strat=="put":pnl=max(K-Se,0)-bs(S,K,T,r,sig,q,"put")
        elif strat=="straddle":pnl=max(Se-K,0)+max(K-Se,0)-bs(S,K,T,r,sig,q,"call")-bs(S,K,T,r,sig,q,"put")
        else:pnl=0
        pnls.append(pnl)
    return np.array(pnls),Sf

# SIDEBAR
with st.sidebar:
    st.write("**PRICER**")
    mode=st.selectbox("",["Pricing","Implied Vol","Backtest"],label_visibility="collapsed")
    method="BS"
    if mode=="Pricing":
        method=st.radio("",["BS","MC"],horizontal=True,label_visibility="collapsed")
    st.write("---")
    S=st.number_input("Spot",value=100.0)
    K=st.number_input("Strike",value=100.0)
    T_d=st.number_input("Maturity (days)",value=30,min_value=1)
    r=st.number_input("Rate (%)",value=5.0)/100
    sig=st.number_input("Volatility (%)",value=20.0)/100
    q=st.number_input("Dividend (%)",value=0.0)/100
    opt=st.radio("",["call","put"],horizontal=True,label_visibility="collapsed")
    
    n_sims=50000
    seed=42
    skew_slope=0.10
    skew_conv=0.05
    term_slope=0.02
    
    if mode=="Pricing" and method=="MC":
        st.write("---")
        n_sims=st.selectbox("Simulations",[10000,50000,100000,200000],index=1)
        seed=st.number_input("Seed",value=42,min_value=1)
    
    if mode=="Implied Vol":
        st.write("---")
        skew_slope=st.slider("Skew slope",-0.50,0.50,0.10,0.01)
        skew_conv=st.slider("Skew convexity",0.00,0.50,0.05,0.01)
        term_slope=st.slider("Term slope",-0.20,0.20,0.02,0.01)
    
    if mode=="Backtest":
        st.write("---")
        strat=st.selectbox("Strategy",["call","put","straddle"])
        nsim=st.selectbox("Simulations",[500,1000,2000],index=1)

T=T_d/365

# MAIN
if mode=="Pricing":
    st.markdown("# " + opt.upper())
    
    if method=="BS":
        p=bs(S,K,T,r,sig,q,opt)
        g=greeks(S,K,T,r,sig,q,opt)
        se=None
        paths=None
    else:
        p,se,paths=mc(S,K,T,r,sig,q,opt,n_sims,seed)
        g=greeks(S,K,T,r,sig,q,opt)
    
    be=K+p if opt=="call" else K-p
    intr=max(S-K,0) if opt=="call" else max(K-S,0)
    d2=(np.log(S/K)+(r-q-0.5*sig**2)*T)/(sig*np.sqrt(T))
    prob_itm=norm.cdf(d2) if opt=="call" else norm.cdf(-d2)
    
    c1,c2=st.columns(2)
    with c1:
        st.write(f"Price = {p:.4f}")
        st.write(f"Break-even = {be:.2f}")
        st.write(f"P(ITM) = {prob_itm*100:.1f}%")
        st.write(f"Time value = {p-intr:.4f}")
        if se:st.write(f"SE = ±{se:.4f}")
    with c2:
        st.write(f"Delta = {g['delta']:.4f}")
        st.write(f"Gamma = {g['gamma']:.5f}")
        st.write(f"Vega = {g['vega']:.4f}")
        st.write(f"Theta = {g['theta']:.4f}")
        st.write(f"Rho = {g['rho']:.4f}")
    
    st.write("")
    fig,ax=plt.subplots(figsize=(6,2.5))
    Sr=np.linspace(S*0.7,S*1.3,150)
    pnl=np.maximum(Sr-K,0)-p if opt=="call" else np.maximum(K-Sr,0)-p
    ax.fill_between(Sr,pnl,0,where=pnl>=0,alpha=0.25,color="#2e7d32")
    ax.fill_between(Sr,pnl,0,where=pnl<0,alpha=0.25,color="#c62828")
    ax.plot(Sr,pnl,color="#333",lw=1)
    ax.axhline(0,color="#999",lw=0.5)
    ax.axvline(K,color="#1565c0",lw=0.7,ls="--")
    ax.axvline(be,color="#2e7d32",lw=0.7,ls="--")
    ax.set_xlim(S*0.7,S*1.3)
    ax.set_xlabel("Spot")
    ax.set_ylabel("P&L")
    ax.grid(True,alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig,use_container_width=False)
    plt.close()
    
    if paths is not None:
        fig,ax=plt.subplots(figsize=(6,2))
        ax.hist(paths,bins=30,color="#1565c0",alpha=0.6,edgecolor="white",lw=0.3)
        ax.axvline(K,color="#c62828",lw=0.7,ls="--")
        ax.set_xlabel("S(T)")
        ax.grid(True,alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig,use_container_width=False)
        plt.close()

elif mode=="Implied Vol":
    st.markdown("# IMPLIED VOL")
    
    iv_skew=skewed_iv(sig,S,K,T,skew_slope,skew_conv,term_slope)
    price_flat=bs(S,K,T,r,sig,q,opt)
    price_skew=bs(S,K,T,r,iv_skew,q,opt)
    gap=price_skew-price_flat
    
    c1,c2=st.columns(2)
    with c1:
        st.write(f"Flat σ = {sig*100:.1f}%")
        st.write(f"Skew IV = {iv_skew*100:.1f}%")
        st.write(f"IV shift = {(iv_skew-sig)*100:+.2f}%")
    with c2:
        st.write(f"Price (flat) = {price_flat:.4f}")
        st.write(f"Price (skew) = {price_skew:.4f}")
        st.write(f"Gap = {gap:+.4f}")
    
    st.write("")
    fig,ax=plt.subplots(figsize=(6,2.5))
    strikes=np.linspace(S*0.7,S*1.3,30)
    ivs=[skewed_iv(sig,S,k,T,skew_slope,skew_conv,term_slope)*100 for k in strikes]
    ax.plot([k/S for k in strikes],ivs,color="#1565c0",lw=1,marker="o",ms=3)
    ax.axhline(sig*100,color="#c62828",lw=0.7,ls="--")
    ax.axvline(1.0,color="#999",lw=0.5,ls=":")
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("IV (%)")
    ax.grid(True,alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig,use_container_width=False)
    plt.close()
    
    fig,ax=plt.subplots(figsize=(6,2))
    mats=np.linspace(max(T,7/365),min(T*3,2.0),20)
    ivs_term=[skewed_iv(sig,S,K,m,skew_slope,skew_conv,term_slope)*100 for m in mats]
    ax.plot([m*365 for m in mats],ivs_term,color="#1565c0",lw=1,marker="o",ms=3)
    ax.axhline(sig*100,color="#c62828",lw=0.7,ls="--")
    ax.axvline(T_d,color="#999",lw=0.5,ls=":")
    ax.set_xlabel("Maturity (days)")
    ax.set_ylabel("IV (%)")
    ax.grid(True,alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig,use_container_width=False)
    plt.close()

else:
    st.markdown("# " + strat.upper())
    
    pnls,Sf=backtest(strat,S,K,T,r,sig,q,T_d,nsim)
    
    c1,c2=st.columns(2)
    with c1:
        st.write(f"Avg P&L = {pnls.mean():.2f}")
        st.write(f"Win rate = {(pnls>0).mean()*100:.0f}%")
        st.write(f"Max = {pnls.max():.2f}")
        st.write(f"Min = {pnls.min():.2f}")
    with c2:
        pcts=np.percentile(pnls,[5,25,50,75,95])
        st.write(f"P5 = {pcts[0]:.2f}")
        st.write(f"P25 = {pcts[1]:.2f}")
        st.write(f"P50 = {pcts[2]:.2f}")
        st.write(f"P75 = {pcts[3]:.2f}")
        st.write(f"P95 = {pcts[4]:.2f}")
    
    st.write("")
    fig,ax=plt.subplots(figsize=(6,2.2))
    ax.hist(pnls[pnls>=0],bins=25,color="#2e7d32",alpha=0.5,edgecolor="white",lw=0.3)
    ax.hist(pnls[pnls<0],bins=25,color="#c62828",alpha=0.5,edgecolor="white",lw=0.3)
    ax.axvline(0,color="#333",lw=0.5)
    ax.axvline(pnls.mean(),color="#1565c0",lw=0.7,ls="--")
    ax.set_xlabel("P&L")
    ax.grid(True,alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig,use_container_width=False)
    plt.close()
    
    fig,ax=plt.subplots(figsize=(6,2))
    ax.scatter(Sf[pnls>=0],pnls[pnls>=0],s=4,alpha=0.35,color="#2e7d32")
    ax.scatter(Sf[pnls<0],pnls[pnls<0],s=4,alpha=0.35,color="#c62828")
    ax.axhline(0,color="#333",lw=0.5)
    ax.axvline(K,color="#1565c0",lw=0.7,ls="--")
    ax.set_xlabel("S(T)")
    ax.set_ylabel("P&L")
    ax.grid(True,alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig,use_container_width=False)
    plt.close()

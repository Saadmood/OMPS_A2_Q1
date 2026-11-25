"""Publication-sized figures, generated directly from the solved Q1 results."""
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

COLORS = {"R":"#626D78", "I":"#C06C16", "C":"#007C83"}
STYLES = {"R":"--", "I":"-.", "C":"-"}
CURVES = {"R":r"$\mathcal{R}$", "I":r"$\mathcal{I}$", "C":r"$\mathcal{C}$"}
SLOPES = ["#007C83", "#396AB1", "#C06C16", "#9C4668"]

plt.rcParams.update({
    "font.family":"DejaVu Sans", "font.size":9.5,
    "axes.labelsize":9.5, "axes.titlesize":10.5, "axes.titleweight":"semibold",
    "legend.fontsize":8.5, "xtick.labelsize":8.5, "ytick.labelsize":8.5,
    "axes.spines.top":False, "axes.spines.right":False,
    "axes.grid":True, "grid.color":"#DDE3E7", "grid.linewidth":.65,
    "axes.axisbelow":True, "savefig.facecolor":"white", "figure.facecolor":"white",
    "pdf.fonttype":42, "ps.fonttype":42,
})


def _save(fig, folder, name):
    folder = Path(folder); folder.mkdir(exist_ok=True,parents=True)
    fig.savefig(folder / (name+".pdf"), bbox_inches="tight", pad_inches=.06)
    fig.savefig(folder / (name+".png"), dpi=180, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def _hour_axis(ax):
    ax.set_xlim(-.5,23.5)
    ax.set_xticks([0,3,6,9,12,15,18,21,23])
    ax.set_xlabel("Hour of day")


def _span_hours(ax, mask, color="#B9C4CB", alpha=.18):
    for t in np.flatnonzero(mask):
        ax.axvspan(t-.5,t+.5,color=color,alpha=alpha,lw=0,zorder=0)


def input_profiles(base, folder):
    """Day-ahead prices, full tariffs, and per-appliance input profiles."""
    com=base.com; p=com.energy_price; h=np.arange(len(p)); ref=com.members[0].reference_load
    pv=next(m.pv_available for m in com.members if m.has_pv)
    fig,ax=plt.subplots(2,1,figsize=(7.0,3.45),sharex=True,layout="constrained")
    ax[0].step(h,p+com.import_tariff,where="mid",ls="--",lw=1.5,color="#657783",label="Full import price")
    ax[0].step(h,p,where="mid",color="#174A72",lw=2,label="Day-ahead price")
    ax[0].step(h,p-com.export_tariff,where="mid",ls=":",lw=1.8,color="#657783",label="Full export price")
    ax[0].set_ylabel("Price [DKK/kWh]"); ax[0].legend(ncol=3,loc="upper left",frameon=False)
    ax[0].set_ylim(.25,3.35)
    ax[1].step(h,ref,where="mid",lw=2,color="#174A72",label="Reference load (one member)")
    ax[1].step(h,pv,where="mid",lw=1.8,color="#D09227",label="PV available (one PV unit)")
    ax[1].set_ylabel("Power [kW]"); ax[1].legend(ncol=2,loc="upper left",frameon=False)
    ax[1].set_ylim(0,3.05); _hour_axis(ax[1]); _save(fig,folder,"input_profiles")


def base_dispatch(base, folder):
    """See congestion removed, and where central dispatch moves each slope class."""
    com=base.com; h=np.arange(24); cap=com.contract.capacity_limit_kW
    fig,axes=plt.subplots(2,1,figsize=(7.0,4.3),gridspec_kw={"height_ratios":[1.6,1]},
                          sharex=True,layout="constrained")
    ax=axes[0]; _span_hours(ax,base.R.aggregate_import>cap+1e-7)
    for s in (base.R,base.I,base.C):
        ax.step(h,s.aggregate_import,where="mid",color=COLORS[s.mode],ls=STYLES[s.mode],lw=2,label=CURVES[s.mode])
    ax.step(h,cap,where="mid",color="#1C2830",ls=":",lw=1.6,label="DSO cap")
    ax.set_ylabel("Metered import [kW]"); ax.set_ylim(0,31)
    ax.legend(ncol=4,loc="upper left",frameon=False)
    ax.set_title("(a) The shared meter eliminates the base-case overshoot",loc="left")
    ax=axes[1]
    changes=np.array([sum(base.C.members[m.consumer_id].load-m.reference_load for m in com.members if m.slope_class==k)
                      for k in (1,2,3,4)])
    lim=max(abs(changes).max(),1.)
    im=ax.imshow(changes,aspect="auto",extent=[-.5,23.5,3.5,-.5],cmap="RdBu_r",
                 norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),interpolation="nearest")
    ax.set_yticks(np.arange(4),["L1 (0.25)","L2 (0.50)","L3 (1.00)","L4 (2.00)"])
    ax.grid(False); ax.set_title("(b) Central load deviation by disutility class",loc="left")
    cb=fig.colorbar(im,ax=ax,orientation="vertical",pad=.018,shrink=.94)
    cb.set_label("Load change [kW]")
    _hour_axis(ax); _save(fig,folder,"base_dispatch")


def settlement_prices(base, folder):
    """Uniform hourly internal price against the import/export marginal tariffs."""
    c,com=base.C,base.com; p=com.energy_price; h=np.arange(24)
    a=(1-c.beta)*com.import_tariff; b=(1-c.beta)*com.export_tariff
    fig,ax=plt.subplots(figsize=(7.0,2.85),layout="constrained")
    _span_hours(ax,c.cap_dual[0]>1e-7,color="#9DD4CC",alpha=.28)
    ax.step(h,p+a,where="mid",color="#657783",ls="--",lw=1.55,label="Discounted import price")
    ax.step(h,p-b,where="mid",color="#657783",ls=":",lw=1.7,label="Discounted export price")
    ax.step(h,c.price[0],where="mid",color=COLORS["C"],lw=2.2,label=r"Internal price $\lambda_t$")
    ax.axhline(next(m.pv_marginal_cost for m in com.members if m.has_pv),color="#C06C16",lw=1.1,ls="-.",label="PV marginal cost")
    ax.set_ylim(.5,5.9); ax.set_ylabel("Price [DKK/kWh]"); _hour_axis(ax)
    fig.legend(loc="outside upper center",ncol=2,frameon=False)
    _save(fig,folder,"settlement_prices")


def contract_sweeps(discount,caps,penalties,thresholds,folder):
    """One panel pair per experiment; identical references and model colours."""
    fig,axes=plt.subplots(3,2,figsize=(7.2,7.15),layout="constrained")
    columns=[(discount,"beta",100.,"Tariff discount [%]",75.,"Discount",
              {k: None if x is None else 100*x for k,x in thresholds["discount_break_even"].items()}),
             (caps,"cap_kW",1.,"Feeder cap [kW]",9.39,"Cap tightness",thresholds["cap_kW_break_even"]),
             (penalties,"alpha_DKK_per_kWh",1.,"Penalty [DKK/kWh] (log scale)",10.,"Penalty",
              thresholds["penalty_cost_break_even"])]
    for r,(frame,col,scale,xlabel,x0,heading,roots) in enumerate(columns):
        for j,(field,ylabel) in enumerate((("gain_vs_R_DKK","Gain vs R [DKK/day]"),
                                            ("exceedance_kWh","Exceedance [kWh/day]"))):
            ax=axes[r,j]
            for m in ("R","I","C"):
                a=frame[frame.model==m].sort_values(col)
                ax.plot(a[col]*scale,a[field],color=COLORS[m],ls=STYLES[m],lw=1.85,label=CURVES[m])
            ax.axvline(x0,color="#89979F",lw=1.05,ls=":",zorder=0)
            if j==0:
                for m,x in roots.items():
                    if x is not None:
                        ax.scatter([x],[0],s=25,color=COLORS[m],edgecolor="white",zorder=5,lw=.4)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            if col=="alpha_DKK_per_kWh":
                ax.set_xscale("log"); ax.set_xticks([.15,.3,.5,1,2,4,10],labels=["0.15","0.3","0.5","1","2","4","10"])
            if j==0: ax.set_title(f"{heading}: community",loc="left")
            else: ax.set_title(f"{heading}: DSO",loc="left")
    handles=[Line2D([0],[0],color=COLORS[m],ls=STYLES[m],lw=2,label=CURVES[m]) for m in ("R","I","C")]
    handles.append(Line2D([0],[0],color="#89979F",ls=":",lw=1.1,label="Base term"))
    fig.legend(handles=handles,loc="outside upper center",ncol=4,frameon=False)
    _save(fig,folder,"contract_sweeps")


def cap_merit_order(caps, folder):
    """Show the additional demand response from progressively stiffer loads."""
    df=caps[caps.model=="C"].sort_values("cap_kW")
    fig,ax=plt.subplots(figsize=(7.0,2.5),layout="constrained")
    for k in (1,2,3,4):
        ax.plot(df.cap_kW,df[f"L{k}_shift_kWh"],color=SLOPES[k-1],lw=1.9,label=f"L{k} ({[.25,.5,1.,2.][k-1]:g})")
    ax.axvline(9.39,color="#89979F",ls=":",lw=1.2,label="Base cap")
    viol=df[df.exceedance_kWh>1e-7]
    if len(viol): ax.axvspan(df.cap_kW.min(),viol.cap_kW.max(),color="#B9C4CB",alpha=.25,lw=0)
    ax.set_xlabel("Feeder cap [kW]"); ax.set_ylabel("Energy shifted [kWh/day]"); ax.set_ylim(bottom=0)
    ax.legend(loc="upper right",ncol=3,frameon=False)
    _save(fig,folder,"cap_merit_order")

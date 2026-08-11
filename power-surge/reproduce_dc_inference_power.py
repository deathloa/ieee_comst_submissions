"""
=============================================================================
Reproducibility Script: Electrical Impact of Data Centers for AI Inference
=============================================================================
Paper: "Power Surge: A Survey of the Electrical Impact of AI Inference in
        Modern Data Centers"
Target Journal: IEEE Communications Surveys and Tutorials

Usage:
    pip install numpy pandas matplotlib scipy tabulate
    python reproduce_dc_inference_power.py

All figures and tables generated match those in the submitted manuscript.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

# ── Colour palette (colour-blind safe) ──────────────────────────────────────
C = {
    "blue":   "#1f77b4",
    "orange": "#ff7f0e",
    "green":  "#2ca02c",
    "red":    "#d62728",
    "purple": "#9467bd",
    "brown":  "#8c564b",
    "grey":   "#7f7f7f",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

OUTPUT_DIR = "."

# ============================================================================
# SECTION 1 – Global Data Center Electricity Consumption (TWh)
# Source: IEA Energy and AI Report 2025, LBNL, EPRI
# ============================================================================

def figure1_global_dc_consumption():
    """
    Fig. 1 — Global data center electricity consumption (TWh), 2015-2030
    with AI-specific workload decomposition.
    """
    years_hist  = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    total_hist  = np.array([196,  213,  228,  255,  270,  284,  318,  365,  395,  415])   # TWh  (IEA 2025)
    ai_hist     = np.array([2,    4,    7,    13,   20,   30,   48,   65,   80,   85])    # TWh  (IEA / EPRI estimates)

    years_proj  = np.array([2025, 2026, 2027, 2028, 2029, 2030])
    total_base  = np.array([480,  560,  660,  760,  860,  945])   # IEA Base Case
    total_high  = np.array([520,  640,  780,  940, 1110, 1300])   # High-growth scenario
    ai_proj     = np.array([140,  210,  310,  420,  540,  660])   # AI share projection

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: total consumption + scenarios
    ax = axes[0]
    ax.fill_between(years_proj, total_base, total_high,
                    alpha=0.18, color=C["blue"], label="Projection range")
    ax.plot(years_hist, total_hist, "o-",  color=C["blue"],   lw=2,   label="Historical total")
    ax.plot(years_proj, total_base, "s--", color=C["blue"],   lw=1.8, label="IEA base case 2030")
    ax.plot(np.concatenate([years_hist, years_proj]),
            np.concatenate([ai_hist, ai_proj]),
            "^-", color=C["orange"], lw=2, label="AI workloads")
    ax.axvline(2024.5, color="grey", lw=1, ls=":")
    ax.text(2024.7, 20, "Projections →", fontsize=8, color="grey")
    ax.set_xlabel("Year")
    ax.set_ylabel("Electricity Consumption (TWh)")
    ax.set_title("Global Data Center Electricity\nConsumption and AI Share")
    ax.legend(fontsize=8)
    ax.set_xlim(2014.5, 2030.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

    # Right: AI share percentage
    ax2 = axes[1]
    share_hist = ai_hist / total_hist * 100
    share_proj = ai_proj / total_base * 100
    all_years  = np.concatenate([years_hist, years_proj])
    all_share  = np.concatenate([share_hist, share_proj])
    ax2.plot(years_hist, share_hist, "o-",  color=C["green"], lw=2, label="Historical AI share")
    ax2.plot(years_proj, share_proj, "s--", color=C["green"], lw=2, label="Projected AI share")
    ax2.axhline(50, color=C["red"], ls=":", lw=1)
    ax2.text(2015, 51.5, "50 % threshold", color=C["red"], fontsize=8)
    ax2.axvline(2024.5, color="grey", lw=1, ls=":")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("AI Workload Share of DC Power (%)")
    ax2.set_title("AI Share of Data Center\nElectricity Consumption")
    ax2.legend(fontsize=8)
    ax2.set_xlim(2014.5, 2030.5)
    ax2.set_ylim(0, 75)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_global_dc_consumption.png", bbox_inches="tight")
    plt.close()
    print("✓  Fig. 1 saved: fig1_global_dc_consumption.png")

    # Print table
    print("\n  Table 1-A: Historical Global DC Electricity (TWh)")
    df = pd.DataFrame({"Year": years_hist, "Total DC (TWh)": total_hist,
                       "AI Workloads (TWh)": ai_hist,
                       "AI Share (%)": (share_hist).round(1)})
    print(df.to_string(index=False))


# ============================================================================
# SECTION 2 – Inference vs Training Power Split
# Source: MLCommons Inference Benchmark, EPRI, Goldman Sachs AI Infrastructure
# ============================================================================

def figure2_inference_vs_training():
    """
    Fig. 2 — Inference vs. training power split over time.
    Inference dominates at production scale (80-90 % of compute cycles).
    """
    years = np.arange(2019, 2031)

    # Training fraction decreasing as models deployed at scale
    train_pct = np.array([35, 32, 28, 25, 22, 20, 18, 16, 14, 13, 12, 11])
    infer_pct = 100 - train_pct

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(years, train_pct, infer_pct,
                 labels=["Training", "Inference"],
                 colors=[C["orange"], C["blue"]], alpha=0.85)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of AI Compute Power (%)")
    ax.set_title("Training vs. Inference Share of AI Data Center Compute Power")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(2019, 2030)
    ax.set_ylim(0, 100)

    # Annotate the inference band midpoint for 2024
    idx_2024 = list(years).index(2024)
    ax.annotate(f"Inference ≈ {infer_pct[idx_2024]}% (2024)",
                xy=(2024, 50), xytext=(2025.5, 45),
                arrowprops=dict(arrowstyle="->", color="white"),
                color="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_inference_vs_training.png", bbox_inches="tight")
    plt.close()
    print("✓  Fig. 2 saved: fig2_inference_vs_training.png")


# ============================================================================
# SECTION 3 – GPU Power Profile: H100 vs A100 vs Blackwell (B200)
# Source: NVIDIA datasheets; MLPerf Inference 4.1 (2024); Epoch AI
# ============================================================================

def figure3_gpu_power_profile():
    """
    Fig. 3 — GPU TDP vs. throughput efficiency (tokens/J) for key inference chips.
    """
    gpus = {
        "A100 SXM (80GB)": {"tdp_w": 400,  "tok_per_s": 150,  "year": 2020},
        "H100 SXM (80GB)": {"tdp_w": 700,  "tok_per_s": 600,  "year": 2022},
        "H200 SXM":        {"tdp_w": 700,  "tok_per_s": 900,  "year": 2024},
        "B200 SXM":        {"tdp_w": 1000, "tok_per_s": 2200, "year": 2025},
        "MI300X (AMD)":    {"tdp_w": 750,  "tok_per_s": 750,  "year": 2024},
        "TPU v5e (Google)":{"tdp_w": 200,  "tok_per_s": 300,  "year": 2023},
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    names  = list(gpus.keys())
    tdps   = [v["tdp_w"]    for v in gpus.values()]
    toks   = [v["tok_per_s"]for v in gpus.values()]
    eff    = [t / p         for t, p in zip(toks, tdps)]   # tok/W
    colors_g = [C["blue"], C["blue"], C["blue"], C["blue"], C["orange"], C["green"]]

    ax = axes[0]
    bars = ax.barh(names, tdps, color=colors_g, alpha=0.85, edgecolor="white")
    for bar, tdp in zip(bars, tdps):
        ax.text(tdp + 10, bar.get_y() + bar.get_height()/2,
                f"{tdp} W", va="center", fontsize=8)
    ax.set_xlabel("Thermal Design Power (W)")
    ax.set_title("GPU TDP for AI Inference\n(LLM, bf16)")
    ax.set_xlim(0, 1250)

    ax2 = axes[1]
    bars2 = ax2.barh(names, eff, color=colors_g, alpha=0.85, edgecolor="white")
    for bar, e in zip(bars2, eff):
        ax2.text(e + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{e:.2f}", va="center", fontsize=8)
    ax2.set_xlabel("Inference Efficiency (tokens / Watt)")
    ax2.set_title("Inference Energy Efficiency\n(LLaMA-70B class, bf16)")

    # Legend patches
    import matplotlib.patches as mpatches
    ax2.legend(handles=[
        mpatches.Patch(color=C["blue"],   label="NVIDIA"),
        mpatches.Patch(color=C["orange"], label="AMD"),
        mpatches.Patch(color=C["green"],  label="Google"),
    ], fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig3_gpu_power_profile.png", bbox_inches="tight")
    plt.close()
    print("✓  Fig. 3 saved: fig3_gpu_power_profile.png")

    # Print table
    print("\n  Table 3-A: Accelerator Power vs. Inference Throughput")
    df = pd.DataFrame([{"GPU": k, "TDP (W)": v["tdp_w"],
                         "Tokens/s (LLaMA-70B)": v["tok_per_s"],
                         "Tokens/W": round(v["tok_per_s"]/v["tdp_w"], 3),
                         "Year": v["year"]}
                        for k, v in gpus.items()])
    print(df.to_string(index=False))


# ============================================================================
# SECTION 4 – Power Usage Effectiveness (PUE) Trends
# Source: Uptime Institute Global Survey 2024; Google ESR 2024; IEA
# ============================================================================

def figure4_pue_trends():
    """
    Fig. 4 — Industry PUE trend and hyperscaler vs. enterprise comparison.
    """
    years_pue  = np.arange(2007, 2025)
    industry   = np.array([2.50, 2.30, 2.10, 1.98, 1.87, 1.80, 1.75, 1.70,
                            1.65, 1.60, 1.58, 1.56, 1.55, 1.53, 1.52, 1.50, 1.49, 1.47])
    hyperscale = np.array([2.10, 1.90, 1.70, 1.55, 1.45, 1.38, 1.30, 1.25,
                            1.20, 1.18, 1.16, 1.15, 1.13, 1.12, 1.12, 1.11, 1.11, 1.10])
    enterprise = np.array([2.60, 2.45, 2.30, 2.20, 2.10, 2.05, 2.00, 1.95,
                            1.90, 1.85, 1.82, 1.78, 1.75, 1.73, 1.72, 1.70, 1.69, 1.67])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(years_pue, hyperscale, enterprise, alpha=0.12, color=C["blue"])
    ax.plot(years_pue, industry,   "o-", color=C["blue"],   lw=2, label="Industry average")
    ax.plot(years_pue, hyperscale, "s-", color=C["green"],  lw=2, label="Hyperscalers (Google/Microsoft/Meta)")
    ax.plot(years_pue, enterprise, "^-", color=C["orange"], lw=2, label="Enterprise DCs")
    ax.axhline(1.0, color="grey", lw=1, ls="--")
    ax.text(2007.2, 1.01, "Theoretical minimum (PUE = 1.0)", fontsize=8, color="grey")
    ax.set_xlabel("Year")
    ax.set_ylabel("Power Usage Effectiveness (PUE)")
    ax.set_title("Data Center PUE Trends by Tier (2007–2024)")
    ax.legend(fontsize=9)
    ax.set_ylim(0.9, 2.7)
    ax.invert_yaxis() if False else None
    ax.set_xlim(2006.5, 2024.5)

    # Annotate 2024 values
    for label, vals, col in [("Hyperscale 1.10", hyperscale, C["green"]),
                               ("Industry 1.47", industry, C["blue"])]:
        ax.annotate(label, xy=(2024, vals[-1]), xytext=(2022.5, vals[-1] - 0.12),
                    arrowprops=dict(arrowstyle="->", color=col), color=col, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_pue_trends.png", bbox_inches="tight")
    plt.close()
    print("✓  Fig. 4 saved: fig4_pue_trends.png")


# ============================================================================
# SECTION 5 – Power Breakdown Inside an AI Inference Rack
# Source: Open Compute Project (OCP) 2024; ASHRAE TC 9.9
# ============================================================================

def figure5_rack_power_breakdown():
    """
    Fig. 5 — Power breakdown inside a dense AI inference rack (H100/B200).
    """
    components_h100 = {
        "GPU (8× H100 SXM)": 5600,
        "CPU (2× Xeon)":      600,
        "Memory (HBM+DRAM)":  400,
        "NVLink / NVSwitch":  300,
        "Networking (InfiniBand 400G)": 250,
        "Storage (NVMe)":     100,
        "Motherboard/misc.":   80,
        "PSU losses (~8%)":   580,
    }
    components_b200 = {
        "GPU (8× B200)":      8000,
        "CPU (2× Xeon)":       600,
        "Memory (HBM3e+DRAM)": 600,
        "NVLink / NVSwitch":   400,
        "Networking (400G IB)":250,
        "Storage (NVMe)":      100,
        "Motherboard/misc.":    80,
        "PSU losses (~8%)":    800,
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, comp, title in [
        (axes[0], components_h100, "H100 SXM 8-GPU Node"),
        (axes[1], components_b200, "B200 SXM 8-GPU Node"),
    ]:
        labels = list(comp.keys())
        values = list(comp.values())
        total  = sum(values)
        gpu_kw = values[0] / 1000.0   # GPU slice, derived from data (no hardcoded label)
        explode = [0.05 if i == 0 else 0 for i in range(len(labels))]
        cmap   = plt.cm.tab10
        colors_pie = [cmap(i / len(labels)) for i in range(len(labels))]
        wedges, texts, autotexts = ax.pie(
            values, labels=None, autopct=lambda p: f"{p:.1f}%" if p > 4 else "",
            colors=colors_pie, explode=explode, startangle=140,
            pctdistance=0.78, wedgeprops={"edgecolor": "white", "linewidth": 1.2})
        ax.set_title(f"{title}\n({gpu_kw:.1f} kW GPU, {total/1000:.1f} kW total IT)")
        ax.legend(wedges, [f"{l} ({v:,} W)" for l, v in zip(labels, values)],
                  fontsize=6.5, loc="lower left", bbox_to_anchor=(-0.05, -0.25))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig5_rack_power_breakdown.png", bbox_inches="tight")
    plt.close()
    print("✓  Fig. 5 saved: fig5_rack_power_breakdown.png")


# ============================================================================
# SECTION 6 – Per-Query Energy Cost for LLM Inference
# Source: Epoch AI (2025); OpenAI engineering estimates; MLPerf Inference 4.1
# ============================================================================

def figure6_per_query_energy():
    """
    Fig. 6 — Energy per query (Wh) across model sizes and hardware generations.
    """
    models = {
        "GPT-3.5 (175B)":  {"h100_wh": 0.001, "b200_wh": 0.00045, "a100_wh": 0.0025},
        "LLaMA-3-70B":     {"h100_wh": 0.0008,"b200_wh": 0.00032, "a100_wh": 0.0018},
        "GPT-4 class (>500B)": {"h100_wh": 0.003,"b200_wh": 0.0012, "a100_wh": 0.007},
        "Gemini Ultra":    {"h100_wh": 0.0028,"b200_wh": 0.0011, "a100_wh": 0.006},
        "Mistral-7B":      {"h100_wh": 0.00012,"b200_wh": 0.00005,"a100_wh": 0.0003},
        "DeepSeek-R1 (671B MoE)": {"h100_wh": 0.0018,"b200_wh": 0.0007,"a100_wh": 0.004},
    }

    names   = list(models.keys())
    x       = np.arange(len(names))
    w       = 0.25

    a100 = [v["a100_wh"] for v in models.values()]
    h100 = [v["h100_wh"] for v in models.values()]
    b200 = [v["b200_wh"] for v in models.values()]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w,   a100, w, label="A100 SXM", color=C["orange"], alpha=0.85)
    ax.bar(x,       h100, w, label="H100 SXM", color=C["blue"],   alpha=0.85)
    ax.bar(x + w,   b200, w, label="B200 SXM", color=C["green"],  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Energy per Query (Wh, 500-token response)")
    ax.set_title("Per-Query Inference Energy Cost by Model and Hardware Generation")
    ax.legend()
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda val, _: f"{val:.4f}" if val < 0.001 else f"{val:.3f}"))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig6_per_query_energy.png", bbox_inches="tight")
    plt.close()
    print("✓  Fig. 6 saved: fig6_per_query_energy.png")

    print("\n  Table 6-A: Per-Query Energy (Wh, 500-token response)")
    rows = [{"Model": k, "A100 (Wh)": v["a100_wh"],
             "H100 (Wh)": v["h100_wh"], "B200 (Wh)": v["b200_wh"],
             "H100→B200 saving (%)": round((1 - v["b200_wh"]/v["h100_wh"])*100, 1)}
            for k, v in models.items()]
    print(pd.DataFrame(rows).to_string(index=False))


# ============================================================================
# SECTION 7 – Cooling Technology Power Comparison
# Source: ASHRAE TC 9.9; OCP Thermal Working Group; Green Revolution Cooling
# ============================================================================

def figure7_cooling_comparison():
    """
    Fig. 7 — Cooling technology PUE overhead, capacity, and water usage.
    """
    cooling_tech = {
        "Air cooling\n(CRAC/CRAH)":     {"pue_add": 0.55, "max_kw_rack": 15,   "wue_lpm": 5.0},
        "Rear-door\nheat exchanger":    {"pue_add": 0.35, "max_kw_rack": 25,   "wue_lpm": 2.5},
        "Direct liquid\ncooling (DLC)": {"pue_add": 0.15, "max_kw_rack": 80,   "wue_lpm": 1.2},
        "Single-phase\nimmersion":      {"pue_add": 0.05, "max_kw_rack": 200,  "wue_lpm": 0.3},
        "Two-phase\nimmersion":         {"pue_add": 0.03, "max_kw_rack": 250,  "wue_lpm": 0.1},
        "Rear-door +\nchilled water":   {"pue_add": 0.25, "max_kw_rack": 40,   "wue_lpm": 1.8},
    }

    names   = list(cooling_tech.keys())
    pue_add = [v["pue_add"]    for v in cooling_tech.values()]
    max_kw  = [v["max_kw_rack"]for v in cooling_tech.values()]
    wue     = [v["wue_lpm"]    for v in cooling_tech.values()]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, data, ylabel, title, col in [
        (axes[0], pue_add, "PUE Overhead Added",      "Cooling PUE Overhead",       C["blue"]),
        (axes[1], max_kw,  "Max Power Density (kW/rack)","Max Rack Power Density",  C["orange"]),
        (axes[2], wue,     "Water Use (L/min per rack)","Water Usage Effectiveness", C["green"]),
    ]:
        bars = ax.barh(names, data, color=col, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, data):
            ax.text(val + max(data)*0.01, bar.get_y() + bar.get_height()/2,
                    f"{val}", va="center", fontsize=8)
        ax.set_xlabel(ylabel)
        ax.set_title(f"{title}")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig7_cooling_comparison.png", bbox_inches="tight")
    plt.close()
    print("✓  Fig. 7 saved: fig7_cooling_comparison.png")


# ============================================================================
# SECTION 8 – Carbon Intensity and Renewable Energy Matching
# Source: Electricity Maps 2024; Google Carbon-Aware Computing; IEA 2025
# ============================================================================

def figure8_carbon_intensity():
    """
    Fig. 8 — Carbon intensity (gCO₂/kWh) by region and renewable fraction.
    """
    regions = {
        "Iceland":          {"ci": 18,   "re_pct": 99},
        "Norway":           {"ci": 29,   "re_pct": 98},
        "France":           {"ci": 56,   "re_pct": 88},
        "Sweden":           {"ci": 45,   "re_pct": 95},
        "Germany":          {"ci": 385,  "re_pct": 55},
        "UK":               {"ci": 233,  "re_pct": 47},
        "US West (WECC)":   {"ci": 220,  "re_pct": 52},
        "US East (PJM)":    {"ci": 410,  "re_pct": 28},
        "US Texas (ERCOT)": {"ci": 360,  "re_pct": 35},
        "Singapore":        {"ci": 430,  "re_pct": 5},
        "China":            {"ci": 560,  "re_pct": 29},
        "India":            {"ci": 710,  "re_pct": 22},
    }

    names = list(regions.keys())
    ci    = [v["ci"]     for v in regions.values()]
    re    = [v["re_pct"] for v in regions.values()]
    # Sort by CI
    idx   = np.argsort(ci)
    names = [names[i] for i in idx]
    ci    = [ci[i]    for i in idx]
    re    = [re[i]    for i in idx]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    color_bars = [C["green"] if c < 200 else C["orange"] if c < 450 else C["red"] for c in ci]
    bars = ax1.barh(names, ci, color=color_bars, alpha=0.85, edgecolor="white")
    ax1.set_xlabel("Grid Carbon Intensity (gCO₂eq / kWh)")
    ax1.set_title("Grid Carbon Intensity and Renewable Fraction by DC Region (2024)")

    ax2 = ax1.twiny()
    ax2.plot(re, names, "D--", color=C["blue"], lw=1.5, ms=6, label="Renewable %")
    ax2.set_xlabel("Renewable Energy Fraction (%)", color=C["blue"])
    ax2.tick_params(axis="x", labelcolor=C["blue"])
    ax2.set_xlim(0, 110)

    # Legend
    import matplotlib.patches as mpatches
    legend_els = [
        mpatches.Patch(color=C["green"],  label="Low CI (<200 gCO₂/kWh)"),
        mpatches.Patch(color=C["orange"], label="Medium CI (200-450)"),
        mpatches.Patch(color=C["red"],    label="High CI (>450)"),
        plt.Line2D([0],[0], color=C["blue"], marker="D", ls="--", label="Renewable %"),
    ]
    ax1.legend(handles=legend_els, fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig8_carbon_intensity.png", bbox_inches="tight")
    plt.close()
    print("✓  Fig. 8 saved: fig8_carbon_intensity.png")


# ============================================================================
# SECTION 9 – Inference Optimization: Quantization & Batching Energy Gains
# Source: Frantar et al. GPTQ 2023; Dettmers et al. QLoRA 2023; vLLM 2024
# ============================================================================

def figure9_optimization_gains():
    """
    Fig. 9 — Energy reduction from quantization and batching strategies.
    """
    quant_levels = ["FP32", "BF16", "INT8\n(GPTQ)", "INT4\n(QLoRA)", "INT4+\n(AWQ)"]
    energy_rel   = [2.0, 1.0, 0.62, 0.40, 0.37]   # relative to BF16 baseline
    throughput_r = [0.5, 1.0, 1.55, 2.10, 2.30]   # relative throughput
    ppl_delta    = [0.0, 0.0, 0.15, 0.55, 0.35]   # perplexity degradation

    batch_sizes  = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    gpu_util     = [12, 22, 38, 58, 74, 86, 92, 96, 98]    # % GPU utilization
    energy_per_tok_batch = [1.0, 0.55, 0.33, 0.22, 0.17, 0.14, 0.13, 0.12, 0.12]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: quantization
    x    = np.arange(len(quant_levels))
    w    = 0.35
    ax   = axes[0]
    ax.bar(x - w/2, energy_rel, w, label="Relative energy",     color=C["red"],   alpha=0.85)
    ax.bar(x + w/2, throughput_r, w, label="Relative throughput",color=C["blue"],  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(quant_levels, fontsize=9)
    ax.set_ylabel("Relative to BF16 Baseline")
    ax.set_title("Quantization: Energy vs.\nThroughput (LLaMA-2-70B)")
    ax.legend(fontsize=9)
    ax.axhline(1.0, color="grey", lw=1, ls="--")

    # Right: batching
    ax2  = axes[1]
    ax2b = ax2.twinx()
    ax2.plot(batch_sizes, energy_per_tok_batch, "o-", color=C["red"],  lw=2, label="Energy/token (rel.)")
    ax2b.plot(batch_sizes, gpu_util, "s--", color=C["blue"], lw=2, label="GPU utilization (%)")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Relative Energy per Token", color=C["red"])
    ax2b.set_ylabel("GPU Utilization (%)",       color=C["blue"])
    ax2.tick_params(axis="y", labelcolor=C["red"])
    ax2b.tick_params(axis="y", labelcolor=C["blue"])
    ax2.set_title("Dynamic Batching:\nEnergy Efficiency vs. GPU Utilization")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig9_optimization_gains.png", bbox_inches="tight")
    plt.close()
    print("✓  Fig. 9 saved: fig9_optimization_gains.png")


# ============================================================================
# SECTION 10 – Network Power in AI Data Centers
# Source: Meta AI Research; Google NeurIPS 2023; Arista 400G specs
# ============================================================================

def figure10_network_power():
    """
    Fig. 10 — Network power as fraction of total DC power.
    """
    years_net = np.arange(2018, 2031)
    net_pct   = np.array([5.0, 5.2, 5.5, 6.0, 6.8, 7.5, 8.5, 9.5, 10.8, 12.0, 13.5, 15.0, 16.5])
    spine_pct = net_pct * 0.45   # spine / core switching
    tor_pct   = net_pct * 0.30   # top-of-rack
    nvlink_pct= net_pct * 0.25   # intra-node (NVLink/NVSwitch)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(years_net, spine_pct, tor_pct, nvlink_pct,
                 labels=["Spine/core switching", "Top-of-rack (TOR)", "Intra-node (NVLink/PCIe)"],
                 colors=[C["blue"], C["orange"], C["green"]], alpha=0.85)
    ax.axvline(2024.5, ls=":", color="grey", lw=1)
    ax.text(2024.7, 1, "Projections →", fontsize=8, color="grey")
    ax.set_xlabel("Year")
    ax.set_ylabel("Network Power as % of Total DC Power")
    ax.set_title("Network Infrastructure Power Fraction in AI Data Centers")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(2017.5, 2030.5)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig10_network_power.png", bbox_inches="tight")
    plt.close()
    print("✓  Fig. 10 saved: fig10_network_power.png")


# ============================================================================
# SECTION 11 – Summary Comparison Table
# ============================================================================

def table_summary():
    """
    Table II — Summary of key metrics for AI inference data centers.
    """
    print("\n" + "="*78)
    print("  TABLE II — Summary of Key Electrical Metrics for AI Inference Data Centers")
    print("="*78)
    data = [
        ["Global DC electricity (2024)",        "≈415 TWh",       "IEA 2025"],
        ["US DC electricity (2024)",             "183 TWh (4% US grid)", "LBNL/EPRI 2024"],
        ["AI share of DC power (2024)",          "10–20%",         "EPRI 2024"],
        ["Projected global DC power (2030)",     "945 TWh (IEA base)", "IEA 2025"],
        ["Inference share of AI compute",        "80–90%",         "MLCommons 2024"],
        ["H100 8-GPU node IT power",             "~7.9 kW (5.6 kW GPU)", "Fig. 5 / NVIDIA DS"],
        ["B200 8-GPU node IT power",             "~10.8 kW (8.0 kW GPU)", "Fig. 5 / NVIDIA DS"],
        ["Energy per GPT-4 query (H100)",        "~3 mWh",         "Epoch AI 2025"],
        ["Energy per ChatGPT query",             "0.3–0.34 Wh",    "OpenAI / Epoch"],
        ["Average industry PUE (2024)",          "1.47",           "Uptime Inst. 2024"],
        ["Hyperscaler average PUE (2024)",       "1.10–1.15",      "Google ESR 2024"],
        ["Two-phase immersion PUE overhead",     "+0.03",          "GRC / OCP 2024"],
        ["Network power fraction (2024)",        "7–10% of rack",  "Meta / Google"],
        ["INT4 quantization energy saving",      "~60% vs BF16",   "AWQ / GPTQ 2024"],
        ["Optimal batch-size energy saving",     "~85% vs batch=1","vLLM benchmarks"],
    ]
    df = pd.DataFrame(data, columns=["Metric", "Value", "Source"])
    print(df.to_string(index=False))
    print("="*78 + "\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Reproducibility Script: DC Inference Electrical Impact")
    print("  IEEE Communications Surveys and Tutorials — Draft 2025")
    print("="*60 + "\n")

    figure1_global_dc_consumption()
    figure2_inference_vs_training()
    figure3_gpu_power_profile()
    figure4_pue_trends()
    figure5_rack_power_breakdown()
    figure6_per_query_energy()
    figure7_cooling_comparison()
    figure8_carbon_intensity()
    figure9_optimization_gains()
    figure10_network_power()
    table_summary()

    print("\n✅  All figures and tables reproduced.")
    print("   Files saved to current directory.")
    print("   Embed PNGs directly into the manuscript figures.\n")

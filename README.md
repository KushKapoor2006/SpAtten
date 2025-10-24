# SpAtten — README

## Project overview

SpAtten explores **speculative token pruning with lookahead** for attention-heavy Transformer stacks, combining a Python Monte‑Carlo simulator and a verification-oriented Verilog RTL model. The simulator measures end-to-end speedups when a predictor speculatively fetches V vectors for next-layer tokens while current-layer AV compute runs, and compares a baseline (no lookahead) against a pipelined lookahead design.

This repository contains:

* `simulation/main_sim.py` — Python Monte Carlo simulator that models baseline and pipelined behavior, generates Gantt charts, histograms, and a precision/recall sweep.
* `simulation/results/` — Generated figures from the simulator (Gantt, histograms, contour, speedup plot, etc.).
* `hardware/` — Verilog verification RTL (module.v, tb.v) implementing the simulator math for cycle-based validation (not synthesizable out-of-the-box).
* `AttentionArchitecture.pdf` — Reference architecture and notes (user-supplied).

This README documents the configuration that produced the **~69.4%** mean speedup observed for the optimistic CV-demo configuration and explains how to reproduce and interpret the outputs.

---

## File structure

```
SPATTEN/
├─ hardware/
│  ├─ module.v
│  └─ tb.v
├─ simulation/
│  ├─ main_sim.py
│  └─ results/
│     ├─ gantt_P0.995_R0.995_CV_DEMO.png
│     ├─ hiding_hist_P0.995_R0.995_CV_DEMO.png
│     ├─ pr_sweep_contour_CV_DEMO.png
│     ├─ speedup_hist_P0.995_R0.995_CV_DEMO.png
│     └─ wasted_hist_P0.995_R0.995_CV_DEMO.png
└─ AttentionArchitecture.pdf
```


> If your tree differs slightly, use the `CONFIG['simulation']['save_path']` in `main_sim.py` to point outputs to the `simulation/results/` directory shown above.

---

## Important run summary (exact `main_sim.py` console output)

```
--- Hardware Sanity Checks (V11 CV Demo Optimistic) ---
Default Bytes per Token (QKV): 4,608 bytes
Speculative/Corrective Bytes per Token (V): 1,536 bytes
BYTES_PER_CYCLE = 128.0000 bytes / cycle
Transfer cycles per token (QKV) ~= 36.00 cycles
Transfer cycles per token (V) ~= 12.00 cycles
Bandwidth window (approx) = 2,560 bytes
------------------------------------------------------------
Ensured directory exists: simulation/results/

Running Monte Carlo: 50 trials for both baseline & pipelined...

--- Results (Mean ± Std over 50 trials) ---
Predictor Precision=0.995, Recall=0.995
  Metric                       | Baseline             | Pipelined (Lookahead)    
--------------------------------------------------------------------------------
  Total Cycles                 | 6,224,129            | 1905201 ± 85
  DRAM Traffic (MB)            | 10.90                | 13.80 ± 0.01
    -> Wasted Traffic (MB)     | N/A                  | 0.01 ± 0.00
  Avg Hidden Time (cycles)     | N/A                  | 3835 ± 7
  Lookahead Finish Rate (%)    | N/A                  | 100.00 ± 0.00

Net Performance Improvement (mean): 69.39%
Saved Gantt: simulation/results/gantt_P0.995_R0.995_CV_DEMO.png
Saved Histograms.

Running precision/recall sweep (coarse)...
Note: All speedup values in sweep were on one side of zero.
      (All points showed speedup)
Saved precision/recall contour: simulation/results/pr_sweep_contour_CV_DEMO.png

Current Optimistic P=0.995, R=0.995 yields SPEEDUP (69.39%).

All plots saved. Finished.
```

**Headline:** With the optimistic CV-demo hardware config and a near‑perfect predictor (P=R=0.995), lookahead speculation produces a **mean 69.39% reduction in total cycles** over the baseline.

---

## What the plots show (brief)

* **gantt_P0.995_R0.995_CV_DEMO.png** — Layer-level Gantt for the *last trial*. Shows per-layer Fetch/QK/Softmax/AV/Score/Top-K durations; pipelined chart overlays predictor/proxy/top-k and speculative/corrective fetch windows.
* **speedup_hist_P0.995_R0.995_CV_DEMO.png** — Distribution of percent speedup (Monte Carlo sampling around the measured mean/std).
* **wasted_hist_P0.995_R0.995_CV_DEMO.png** — Distribution of wasted DRAM traffic due to false-positive speculative fetches (very small here: ~0.01 MB).
* **hiding_hist_P0.995_R0.995_CV_DEMO.png** — Distribution of hidden time (how much of AV latency was overlapped by lookahead) — mean ~3835 cycles.
* **pr_sweep_contour_CV_DEMO.png** — Sweep of predictor P (x-axis) and R (y-axis) showing net speedup (%) as a contour; for this optimistic config all sampled points gave positive speedup.

---

## How the simulator models the system (short)

* **Compute latencies** are modeled as simple multiplicative forms: e.g., `qk_mult_factor * N * N * H`, `softmax_factor * N * N * H`, `av_mult_factor * N * N * H`, `score_comp_factor * N * N * H`, `proxy_col_sum_factor * N * N`, `top_k_factor * N`.
* **Memory fetch** latency = DRAM latency (cycles) + transfer cycles (bytes_to_fetch / bytes_per_cycle) + saturation window extra cycles if bytes_to_fetch > BW_WINDOW_BYTES.
* **Baseline**: fetch full QKV (Q,K,V) for `num_tokens`; execute QK/Softmax/AV; then blocking score computation and Top-K stall; prune tokens.
* **Pipelined / Lookahead**: in parallel with AV, predictor + proxy + top-k are executed for next-layer candidates; the simulator issues speculative fetch of V for predicted K while AV runs; corrective fetch (if FN) finishes after.
* **Monte Carlo**: randomness via binomial sampling of true next-K given predictor recall; trials aggregate mean/std.

---

## Key assumptions and simplifications

1. **Optimistic memory & compute**: Bandwidth = 1024 Gb/s and clock = 1 GHz are extremely optimistic — intended to showcase the mechanism rather than represent a conservative HW point.
2. **Near-perfect predictor scenario**: P=R=0.995 demonstrates upper-bound benefits. Lower-quality predictors will reduce speedup and increase wasted fetch traffic.
3. **Pruning model**: Tokens pruned deterministically by `token_pruning_rate` per layer (20% per layer) — this is a simplification of actual token selection dynamics.
4. **Verification-oriented Verilog**: the RTL in `hardware/` is purposely written for functional verification; it computes the same numeric metrics but uses heavy combinational arithmetic and single-cycle-per-layer semantics. It is not ready for synthesis without pipelining and operator decomposition.

---

## How to reproduce results

1. Install Python3 + numpy + matplotlib.
2. From `project_root/` run:

```bash
python3 simulation/main_sim.py
```

3. Outputs will be placed in `simulation/results/` (or the `save_path` defined in `CONFIG`). Inspect the Gantt, histograms, and contour.

For the Verilog run (functional verification):

```bash
cd hardware
iverilog -g2012 -o spatten_hw_sim module.v tb.v
vvp spatten_hw_sim
```

> Note: Use the Verilog model for numeric checks and behavioral validation; refactor it to pipelined multi-cycle datapath if you need a synthesizable design.

---

## Interpretation of the 69.4% speedup

Under the chosen optimistic hardware and predictor settings, the lookahead pipeline completes predictor+proxy+top-k+speculative fetch **before** AV finishes for every layer (Lookahead Finish Rate = 100%). That means the expensive `score_comp`/`top_k` stall is effectively overlapped by lookahead activity (hidden time ≈ 3835 cycles on average). The combination of near-perfect prediction and very fast speculative fetch (only V, not Q/K) yields a large net reduction in end-to-end cycles despite a small increase in DRAM traffic due to speculation.

---

## Limitations & suggested next steps

* **Synthesis-ready RTL**: break large combinational math into pipelined units; implement multi-cycle division/multiplication or use vendor DSPs; add a config interface (AXI‑Lite) and enable runtime tuning of predictor parameters.
* **Hardware-accurate DRAM model**: consider more realistic DRAM controllers, row-buffer effects, and request queuing which affect latency/bandwidth tradeoffs.
* **Predictor design & cost**: integrate a cycle-accurate predictor model (hardware cost, area, latency) rather than treating predictor cost as a scalar factor.
* **Heterogeneous fetch sizes**: model fetching partial V tiles across DDR bursts and prefetch behavior for more realistic memory traffic.

---

## Contact / authorship

Project: SpAtten
Author: (replace with your name / handle)


Say `generate EVALUATION.md` and I will create it in the canvas.

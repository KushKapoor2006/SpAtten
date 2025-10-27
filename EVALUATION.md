# SpAtten — EVALUATION

This document provides a thorough, reproducible evaluation of the SpAtten experiments and artifacts included in the repository. It collects the simulation parameters, key results, assumptions, limitations, interpretation, and an actionable hardware implementation plan to move from the current verification-oriented Verilog to a synthesizable accelerator.

---

## 1. Executive summary

* Using the optimistic CV-demo configuration in `main_sim.py` (high memory bandwidth, near-perfect predictor P=R=0.995, 12 layers, 512 tokens initial), the Python Monte Carlo simulator reports a **mean net speedup of 69.39%** for the pipelined (lookahead/speculative) design compared to the baseline.
* Important supporting metrics from the run (50 trials):

  * Baseline total cycles (mean): **6,224,129**
  * Pipelined total cycles (mean): **1,905,201 ± 85**
  * DRAM traffic (mean): Baseline **10.90 MB**, Pipelined **13.80 ± 0.01 MB**
  * Wasted DRAM (pipelined): **0.01 MB ± 0.00** (negligible)
  * Average hidden time per layer (pipelined): **3,835 cycles ± 7**
  * Lookahead Finish Rate: **100.00%** (the lookahead path finished before AV in every layer)

Interpretation: under these optimistic hardware & predictor parameters, lookahead speculation reliably hides the blocking score/top-k stall and yields a large end-to-end cycle reduction while incurring small extra DRAM traffic.

---

## 2. Exact configuration used for the reported run

All values are from `main_sim.py` `CONFIG` dictionary and printed derived constants.

### Model & pruning

| Parameter                    |      Value |
| ---------------------------- | ---------: |
| `model.num_layers`           |         12 |
| `model.initial_seq_len`      |        512 |
| `model.initial_heads`        |         12 |
| `pruning.token_pruning_rate` | 0.20 (20%) |

### Simulation

| Parameter                       | Value |
| ------------------------------- | ----: |
| `simulation.monte_carlo_trials` |    50 |
| `simulation.random_seed`        |    42 |

### Memory & hardware latency (CV-demo optimistic)

| Parameter                                    |                  Value |
| -------------------------------------------- | ---------------------: |
| `memory.latency`                             |              20 cycles |
| `memory.bytes_per_token_vecs`                |                      3 |
| `memory.embedding_dim`                       |                    768 |
| `memory.bytes_per_element`                   |                      2 |
| `memory.bandwidth_Gb_per_s`                  |            1024.0 Gb/s |
| `memory.clock_speed_ghz`                     |                1.0 GHz |
| Derived `BYTES_PER_VEC`                      |   768 * 2 = 1536 bytes |
| Derived `DEFAULT_BYTES_PER_TOKEN` (QKV)      | 1536 * 3 = 4,608 bytes |
| Derived `SPEC_CORR_BYTES_PER_TOKEN` (V only) |             1536 bytes |
| Derived `BYTES_PER_CYCLE`                    |      128 bytes / cycle |
| `BW_WINDOW_BYTES` (approx)                   |            2,560 bytes |

### Hardware latency factors (multiplicative)

| Metric                     | Factor value |
| -------------------------- | -----------: |
| `score_comp_factor`        |         0.50 |
| `proxy_col_sum_factor`     |        0.001 |
| `top_k_factor`             |         0.01 |
| `qk_mult_factor`           |         0.10 |
| `softmax_factor`           |       0.0008 |
| `av_mult_factor`           |         0.10 |
| `predictor_compute_factor` |        0.001 |

### Speculation / predictor

| Parameter                            |            Value |
| ------------------------------------ | ---------------: |
| `speculation.predictor_precision`    |            0.995 |
| `speculation.predictor_recall`       |            0.995 |
| `speculation.speculative_fetch_vecs` | 1 (fetch V only) |
| `speculation.corrective_fetch_vecs`  |                1 |

---

## 3. Key outputs & artifacts (files)

The simulator saved figures under `simulation/results/` for the run; these are included in the repository.

* `gantt_P0.995_R0.995_CV_DEMO.png` — Gantt chart (last trial) showing stage timings and lookahead overlays.
* `speedup_hist_P0.995_R0.995_CV_DEMO.png` — Speedup distribution (histogram).
* `wasted_hist_P0.995_R0.995_CV_DEMO.png` — Wasted DRAM traffic histogram.
* `hiding_hist_P0.995_R0.995_CV_DEMO.png` — Hidden time histogram.
* `pr_sweep_contour_CV_DEMO.png` — Contour of net speedup vs predictor precision/recall.

---

## 4. Assumptions & simplifications (complete list)

The simulator and RTL model make several modeling choices designed to stress the lookahead mechanism and keep the study tractable:

1. **Optimistic DRAM bandwidth & latency** — 1024 Gb/s and 20 cycles latency are best-case values for demonstration (not necessarily representative of commodity DDR/LPDDR subsystems). This strongly influences speculative fetch latency being small and lookahead's ability to hide stalls.

2. **Predictor quality near-perfect** — P=R=0.995 is intentionally optimistic to demonstrate maximum potential. In practice predictor quality can vary and must be balanced against hardware cost and added latency.

3. **Speculative fetch size limited to V-only** — Fetching only V vectors per speculative candidate minimizes speculative DRAM overhead and speeds transfers (V-only fetch cycles ≈ 12 cycles vs QKV ≈ 36 cycles per token under the chosen bandwidth).

4. **Token pruning is multiplicative and deterministic** — Each layer prunes tokens by a fixed factor (20% per layer, floor to at least 1). Real pruning strategies choose token sets based on scores; simulation approximates the count.

5. **Lookahead models compute cost as scalar factors** — Predictor/proxy/top-k compute costs are scalar factors multiplied by N/H counts (approximate arithmetic complexity). The predictor costs are intentionally tiny to mimic a near-free predictor (e.g., cheap model) in the CV demo.

6. **Monte Carlo sampling at token-count level** — The Python sim draws binomial samples for true-k sampling (deterministic approximations can appear in the Verilog model).

7. **DRAM model is idealized** — No row-buffer locality, request queueing, or burst alignment is modeled explicitly. The bandwidth window adds a first-order saturation effect.

8. **Verilog RTL is verification-oriented** — it reproduces the arithmetic and cycle counts but uses heavy combinational math and single-cycle-per-layer semantics. It is not a synthesizable accelerator implementation.

---

## 5. Interpretation of results (detailed)

**Why 69.39%?** The large reported speedup arises from three interacting conditions:

1. **High predictor quality** reduces wasted speculative fetches (low FP) and lowers corrective re-fetches (low FN), keeping wasted DRAM traffic nearly zero.
2. **High memory bandwidth / low transfer cycles** make speculative V fetches small relative to compute latencies: fetching a V token costs ≈12 cycles while QKV would cost ≈36 cycles, enabling speculation to be cheap.
3. **Compute/latency balance**: the lookahead compute chain (predictor + proxy + top-k) plus speculative fetch completes *before* AV compute finishes (Lookahead Finish Rate 100%), making the speculative path effectively hide substantial AV/score stall time (hidden time ≈3835 cycles), thus dramatically reducing total cycles.

**Tradeoffs**

* Pipelined lookahead increases DRAM traffic (13.80 MB vs 10.90 MB mean), but the extra traffic is small relative to system capacity in the CV-demo. In low-bandwidth or high-cost DRAM systems, the extra traffic could offset benefits.
* With lower predictor quality, wasted fetches grow and the net speedup declines. The PR-sweep shows that for the chosen HW point most points were still positive, but that will not hold for all HW.

---

## 6. Reproducibility (exact commands)

### Python simulation

```bash
cd simulation
python3 main_sim.py
# Inspect outputs: simulation/results/*.png
```

Ensure Python packages: `numpy`, `matplotlib`, and standard library modules are available.

### Verilog verification run (behavioral)

```bash
cd hardware
iverilog -g2012 -o spatten_sim module.v tb.v
vvp spatten_sim
```

The Verilog model is intended for numeric checks only (verification) and is not optimized for synthesis.

---

## 7. Suggested hardware implementation plan

Below is a concrete, practical plan to convert the verification RTL into a synthesizable, high-performance accelerator.

### 7.1. High-level architecture

* **Control plane (FSM)** — micro-op sequencer that issues datapath operations per layer: FETCH, QK, SOFTMAX, AV, PREDICTOR, PROXY, TOPK, SPEC_FETCH, CORR_FETCH, SCORE.

* **Datapath units (pipelined modules)**:

  * `fetch_engine` — handles byte-to-cycle conversion, burst alignment, and issues DRAM read transactions; multi-cycle with handshake.
  * `latency_estimator` — small combinational/pipelined unit to estimate compute latencies (can be run ahead-of-time in control path).
  * `predictor_unit` — cycle-accurate predictor (could be a small NN or heuristic) with explicit latency and area budget.
  * `proxy_score_unit` — compute approximate column sums / proxy scores (pipelined).
  * `topk_unit` — streaming top-K accelerator for selection (pipelined or multi-pass depending on resources).
  * `av_unit` — AV multiplication engine (matrix/vector) implemented using DSP arrays.

* **Scratch & caches**:

  * small on-chip buffer for speculative V slices (to avoid repeated DRAM roundtrips when speculation succeeds); allocate for predicted_k * vector_size.

### 7.2. Design guidelines & optimizations

* **Replace large single-cycle divides** with multi-cycle dividers or fixed-point scaling by powers-of-two. Consider changing `SCALE` to a power-of-two to allow shift-based scaling.
* **Pipeline multiply-heavy datapath** across several stages, exposing DSPs for multipliers. Keep each pipeline stage’s combinational delay under the target clock period.
* **Implement fetch engine as a request/response unit**: issue speculative fetch requests with low priority and allow hardware to merge/coalesce requests to maximize DRAM throughput.
* **Predictor latency budget**: ensure the predictor + proxy + top-k critical path is purposely bounded to finish before AV in the common case. If necessary, reduce predictor complexity or prefetch more aggressively.
* **Resource-aware top-k**: implement streaming top-K or approximate top-K to bound cost; exact top-K on large N is expensive.
* **Backpressure & correctness**: design pipelines with valid/ready handshakes so corrective fetches can be scheduled without data hazards.

### 7.3. Implementation steps

1. **Bit-width analysis**: calculate maximum values for counters and accumulators for your widest configuration (NUM_LAYERS, initial_seq_len, embedding_dim) and reduce widths to the minimum safe size.
2. **Split the verification combinational logic into pipeline stages**. Start with a 3–6 stage pipeline covering fetch latency computation, predictor/proxy/topk, and AV compute.
3. **Provide synthesis-friendly modules**: implement multi-cycle divide IP for bytes_to_cycle conversion if needed or replace division by constant with multiplication (precompute reciprocal) and shifts.
4. **Add runtime-config registers** (AXI‑Lite or a simple register file) to tune `token_pruning_rate`, `predictor_precision/recall`, `speculative_vecs`, and `bandwidth` for experiments.
5. **Validate against Python golden model**: generate per-layer traces (fetch sizes, start/end cycles) and compare hardware outputs for multiple configurations and seeds.
6. **Map heavy ops to FPGA DSPs/BRAM** and iterate on floorplanning/pipelining to meet timing.

---

## 8. Suggested experiments & ablations

To more fully characterize the design, run the following sweeps (Python sim) and record contour/histograms:

1. **Bandwidth sweep** — vary `memory.bandwidth_Gb_per_s` from conservative to optimistic (e.g., 32, 128, 512, 1024) and observe how speedup and wasted traffic change.
2. **Predictor quality sweep** — finer PR grid in the range [0.7, 0.999] to find break-even curves (already plotted coarsely). Use a denser grid near the interesting transition boundary.
3. **Pruning rate sweep** — vary `pruning.token_pruning_rate` (0.05–0.5) to measure sensitivity to token retention per layer.
4. **Speculative fetch size** — compare fetching V-only vs QKV vs partial V tiles (increase `speculative_fetch_vecs`) to explore bandwidth/accuracy tradeoffs.
5. **Layer/seq_len scaling** — test num_layers ∈ {6,12,24} and seq_len ∈ {128,256,512,1024} to see scaling behaviour.
6. **Predictor latency sensitivity** — adjust `hardware_latency.predictor_compute_factor` to emulate slow predictors and see the impact on Lookahead Finish Rate.

For each sweep, store the full `metrics` arrays and plot heatmaps/contours of Net Speedup and Wasted Traffic.

---

## 9. Limitations and open questions

* **DRAM realism**: request-level effects (row-buffer locality, bank conflicts) will change fetch latency and saturation behavior.
* **Predictor cost & area**: the simulator treats predictor compute as tiny; a hardware predictor with comparable precision may be non-trivial to implement and could introduce latency/area tradeoffs that reduce net benefit.
* **Speculation correctness & recovery**: the model assumes corrective fetches repair false negatives without complex reordering costs — on hardware, reordering and correctness (out-of-order data arrival) require careful design.
* **Energy & area**: this evaluation focuses on cycles and bytes; energy cost of speculative fetches and predictor acceleration must be measured for a full cost/benefit analysis.

---

## 10. Appendices

### A. Exact console log (key excerpt)

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

### B. Reproducibility checklist

* Ensure `main_sim.py` `CONFIG` is unchanged and is in the same folder
* Run the Python script with the same `random_seed` to reproduce identical Monte Carlo draws.
* To compare Verilog vs Python outputs, produce per-layer traces (start/end times) in Python and have the Verilog testbench print the same trace; run comparisons (diff or script) to verify numeric parity.

---

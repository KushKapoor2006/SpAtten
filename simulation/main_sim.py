import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
import json

# --- V11: Highly Optimistic
CONFIG = {
    "simulation": {
        "monte_carlo_trials": 50, # Fewer trials for faster demo runs
        "save_path": "simulation/results/", # Separate dir
        "random_seed": 42,
    },
    "model": {
        "num_layers": 12,
        "initial_seq_len": 512,
        "initial_heads": 12,
    },
    "pruning": {
        "token_pruning_rate": 0.20,
    },
    "hardware_latency": {
        # --- CV DEMO FACTORS ---
        "score_comp_factor": 0.50,       # MASSIVE baseline stall cost
        "proxy_col_sum_factor": 0.001,   # Almost free lookahead proxy
        "top_k_factor": 0.01,            # Almost free lookahead topk
        "qk_mult_factor": 0.10,          # High compute cost
        "softmax_factor": 0.0008,        # Fast softmax
        "av_mult_factor": 0.10,          # High compute cost (hiding spot)
        "predictor_compute_factor": 0.001 # Almost free predictor
    },
    "memory": {
         # --- CV DEMO MEMORY (Ultra Fast) ---
        "latency": 20,                   # Very low latency
        "bytes_per_token_vecs": 3,
        "embedding_dim": 768,
        "bytes_per_element": 2,
        "bandwidth_Gb_per_s": 1024.0,    # Extremely high bandwidth
        "clock_speed_ghz": 1.0,
        "saturation_window_multiplier": 1.0 # Keep simple saturation
    },
    "speculation": {
        # --- CV DEMO PREDICTOR (Near Perfect) ---
        "predictor_precision": 0.995,    # Near perfect
        "predictor_recall": 0.995,       # Near perfect
        "speculative_fetch_vecs": 1,     # Fetch only V
        "corrective_fetch_vecs": 1      # Corrective fetch only V
    }
}

np.random.seed(CONFIG['simulation']['random_seed'])

# --- Derived constants & sanity checks ---
BYTES_PER_VEC = CONFIG['memory']['embedding_dim'] * CONFIG['memory']['bytes_per_element']
DEFAULT_BYTES_PER_TOKEN = BYTES_PER_VEC * CONFIG['memory']['bytes_per_token_vecs']
SPEC_CORR_BYTES_PER_TOKEN = BYTES_PER_VEC * CONFIG['speculation']['speculative_fetch_vecs'] # Same for corrective
BANDWIDTH_BITS_PER_S = CONFIG['memory']['bandwidth_Gb_per_s'] * 1e9
BANDWIDTH_BYTES_PER_S = BANDWIDTH_BITS_PER_S / 8.0
CYCLES_PER_S = CONFIG['memory']['clock_speed_ghz'] * 1e9
# Handle potential division by zero if clock speed is zero
BYTES_PER_CYCLE = BANDWIDTH_BYTES_PER_S / CYCLES_PER_S if CYCLES_PER_S > 0 else 0
MEM_LATENCY_CYCLES = CONFIG['memory']['latency']
MEM_LATENCY_SEC = MEM_LATENCY_CYCLES / CYCLES_PER_S if CYCLES_PER_S > 0 else 0
BW_WINDOW_BYTES = BANDWIDTH_BYTES_PER_S * (MEM_LATENCY_SEC * CONFIG['memory']['saturation_window_multiplier'])

print("--- Hardware Sanity Checks (V11 CV Demo Optimistic) ---")
print(f"Default Bytes per Token (QKV): {DEFAULT_BYTES_PER_TOKEN:,} bytes")
print(f"Speculative/Corrective Bytes per Token (V): {SPEC_CORR_BYTES_PER_TOKEN:,} bytes")
print(f"BYTES_PER_CYCLE = {BYTES_PER_CYCLE:.4f} bytes / cycle")
# Avoid division by zero in print statement
print(f"Transfer cycles per token (QKV) ~= {(DEFAULT_BYTES_PER_TOKEN / BYTES_PER_CYCLE):.2f} cycles" if BYTES_PER_CYCLE > 0 else "N/A (check clock speed)")
print(f"Transfer cycles per token (V) ~= {(SPEC_CORR_BYTES_PER_TOKEN / BYTES_PER_CYCLE):.2f} cycles" if BYTES_PER_CYCLE > 0 else "N/A")
print(f"Bandwidth window (approx) = {BW_WINDOW_BYTES:,.0f} bytes")
print("-" * 60)


class HardwareSimulator:
    def __init__(self, config):
        self.config = config
        self.log = []
        self.total_time = 0.0
        self.metrics = {}

    def _get_latency(self, stage, num_tokens, num_heads):
        N, H = float(num_tokens), float(num_heads)
        if N <= 0 or H <= 0: return 1.0 # Avoid issues with zero tokens/heads
        lat = self.config['hardware_latency']
        val = 0.0
        # Added checks for very large numbers just in case factors are extreme
        try:
            if stage == 'score_comp': val = lat['score_comp_factor'] * N * N * H
            elif stage == 'proxy_col_sum': val = lat['proxy_col_sum_factor'] * N * N
            elif stage == 'top_k': val = lat['top_k_factor'] * N
            elif stage == 'qk_mult': val = lat['qk_mult_factor'] * N * N * H
            elif stage == 'softmax': val = lat['softmax_factor'] * N * N * H
            elif stage == 'av_mult': val = lat['av_mult_factor'] * N * N * H
            elif stage == 'predictor': val = lat['predictor_compute_factor'] * N
            # Prevent overflow or excessively large unrealistic latencies
            if not np.isfinite(val) or val > 1e12: val = 1e12
        except OverflowError:
             val = 1e12 # Cap latency if calculation overflows
        return max(1.0, float(val))


    def _get_fetch_latency(self, num_tokens_to_fetch, num_vecs_override=None):
        if num_tokens_to_fetch <= 0: return 1.0
        num_vecs = num_vecs_override if num_vecs_override is not None else self.config['memory']['bytes_per_token_vecs']
        bytes_per_token_effective = BYTES_PER_VEC * num_vecs
        if bytes_per_token_effective <= 0: return 1.0
        bytes_to_fetch = float(num_tokens_to_fetch) * bytes_per_token_effective
        if BYTES_PER_CYCLE <= 1e-9: return float('inf') # Infinite latency if no bandwidth
        transfer_cycles = bytes_to_fetch / BYTES_PER_CYCLE
        extra_cycles = 0.0
        # Only apply saturation if window > 0
        if BW_WINDOW_BYTES > 0 and bytes_to_fetch > BW_WINDOW_BYTES:
            extra_bytes = bytes_to_fetch - BW_WINDOW_BYTES
            extra_cycles = extra_bytes / BYTES_PER_CYCLE
        latency_val = float(MEM_LATENCY_CYCLES + transfer_cycles + extra_cycles)
        # Cap fetch latency as well
        if not np.isfinite(latency_val) or latency_val > 1e12: latency_val = 1e12
        return max(1.0, latency_val)

    def run(self): raise NotImplementedError
    def _log_event(self, layer_idx, stage_name, start_time, end_time, color):
        duration = float(end_time - start_time)
        if duration > 1e-9: # Log if duration is measurably positive
            self.log.append({
                "layer": f"Layer {layer_idx}", "stage": stage_name,
                "start": float(start_time), "duration": duration,
                "color": color
            })

# --- Baseline and Pipelined Simulators (Logic unchanged from v9) ---
class BaselineSpAttenSimulator(HardwareSimulator):
    def run(self):
        current_time = 0.0
        num_tokens = int(self.config['model']['initial_seq_len'])
        num_heads = int(self.config['model']['initial_heads'])
        total_dram_bytes = 0.0
        for i in range(self.config['model']['num_layers']):
            if num_tokens <= 0: break # Stop if tokens pruned to zero
            lat_fetch = self._get_fetch_latency(num_tokens)
            total_dram_bytes += num_tokens * DEFAULT_BYTES_PER_TOKEN
            self._log_event(i, 'Fetch', current_time, current_time + lat_fetch, 'gray')
            current_time += lat_fetch
            lat_qk = self._get_latency('qk_mult', num_tokens, num_heads)
            lat_softmax = self._get_latency('softmax', num_tokens, num_heads)
            lat_av = self._get_latency('av_mult', num_tokens, num_heads)
            self._log_event(i, 'QK Mult', current_time, current_time + lat_qk, 'coral')
            current_time += lat_qk
            self._log_event(i, 'Softmax', current_time, current_time + lat_softmax, 'gold')
            current_time += lat_softmax
            self._log_event(i, 'AV Mult', current_time, current_time + lat_av, 'salmon')
            current_time += lat_av
            # --- This is now the HUGE stall ---
            lat_score = self._get_latency('score_comp', num_tokens, num_heads)
            lat_topk = self._get_latency('top_k', num_tokens, num_heads)
            self._log_event(i, 'Score Comp (Stall)', current_time, current_time + lat_score, 'skyblue')
            current_time += lat_score
            self._log_event(i, 'Top-K (Stall)', current_time, current_time + lat_topk, 'dodgerblue')
            current_time += lat_topk
            num_tokens = int(max(1, num_tokens * (1 - self.config['pruning']['token_pruning_rate'])))
        self.total_time = current_time
        self.metrics = {"Total Cycles": self.total_time, "DRAM Traffic (MB)": total_dram_bytes / 1e6}
        return self.log, self.metrics


class PipelinedSpAttenSimulator(HardwareSimulator):
    def run(self):
        current_time = 0.0
        num_tokens = int(self.config['model']['initial_seq_len'])
        num_heads = int(self.config['model']['initial_heads'])
        total_dram_bytes = 0.0
        total_wasted_bytes = 0.0
        hidden_times = []
        finish_before_av_count = 0
        speculative_vecs = self.config['speculation']['speculative_fetch_vecs']
        corrective_vecs = self.config['speculation']['corrective_fetch_vecs']
        speculative_bytes_per_token = BYTES_PER_VEC * speculative_vecs
        corrective_bytes_per_token = BYTES_PER_VEC * corrective_vecs

        for i in range(self.config['model']['num_layers']):
            if num_tokens <= 0: break # Stop if tokens pruned to zero
            prec = float(self.config['speculation']['predictor_precision'])
            rec = float(self.config['speculation']['predictor_recall'])
            true_k_next = int(max(1, num_tokens * (1 - self.config['pruning']['token_pruning_rate'])))

            # Clamp recall to avoid issues if true_k_next is very small
            rec_clamped = min(1.0, max(0.0, rec))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                 # Ensure n is non-negative for binomial
                tp = int(np.random.binomial(max(0, true_k_next), rec_clamped))

            predicted_k = int(np.round(tp / prec)) if prec > 1e-9 else num_tokens
            predicted_k = min(predicted_k, num_tokens)
            fp = max(0, predicted_k - tp)
            fn = max(0, true_k_next - tp)

            lat_fetch = self._get_fetch_latency(num_tokens)
            total_dram_bytes += num_tokens * DEFAULT_BYTES_PER_TOKEN
            self._log_event(i, 'Fetch', current_time, current_time + lat_fetch, 'gray')
            current_time += lat_fetch

            lat_qk = self._get_latency('qk_mult', num_tokens, num_heads)
            lat_soft = self._get_latency('softmax', num_tokens, num_heads)
            self._log_event(i, 'QK Mult', current_time, current_time + lat_qk, 'coral')
            self._log_event(i, 'Softmax', current_time + lat_qk, current_time + lat_qk + lat_soft, 'gold')
            parallel_start_time = current_time + lat_qk + lat_soft

            lat_av = self._get_latency('av_mult', num_tokens, num_heads)
            corrective_fetch_penalty = self._get_fetch_latency(fn, num_vecs_override=corrective_vecs)
            path_A_end = parallel_start_time + lat_av + corrective_fetch_penalty

            lat_predictor = self._get_latency('predictor', num_tokens, num_heads)
            lat_proxy = self._get_latency('proxy_col_sum', num_tokens, num_heads)
            lat_topk = self._get_latency('top_k', num_tokens, num_heads)
            lookahead_compute = lat_predictor + lat_proxy + lat_topk
            speculative_fetch_time = self._get_fetch_latency(predicted_k, num_vecs_override=speculative_vecs)
            path_B_end = parallel_start_time + lookahead_compute + speculative_fetch_time

            if (parallel_start_time + lookahead_compute + speculative_fetch_time) <= (parallel_start_time + lat_av):
                finish_before_av_count += 1
            hidden_time = min(lat_av, lookahead_compute + speculative_fetch_time)
            hidden_times.append(hidden_time)

            self._log_event(i, 'AV Mult', parallel_start_time, parallel_start_time + lat_av, 'salmon')
            if corrective_fetch_penalty > 1:
                self._log_event(i, 'Corrective Fetch (FN)', parallel_start_time + lat_av, path_A_end, 'red')

            if i < self.config['model']['num_layers'] - 1:
                pred_start = parallel_start_time
                self._log_event(i + 1, 'Predictor*', pred_start, pred_start + lat_predictor, 'mediumpurple')
                proxy_start = pred_start + lat_predictor
                self._log_event(i + 1, 'Proxy Score*', proxy_start, proxy_start + lat_proxy, 'skyblue')
                topk_start = proxy_start + lat_proxy
                self._log_event(i + 1, 'Top-K*', topk_start, topk_start + lat_topk, 'dodgerblue')
                spec_fetch_start = parallel_start_time + lookahead_compute
                self._log_event(i + 1, f'Spec Fetch ({speculative_vecs}V)*', spec_fetch_start, spec_fetch_start + speculative_fetch_time, 'gray')
                if fp > 0 and predicted_k > 0:
                    wasted_fraction = float(fp) / float(predicted_k)
                    wasted_time_overlay = speculative_fetch_time * wasted_fraction
                    self._log_event(i + 1, 'Wasted Fetch (FP)', spec_fetch_start, spec_fetch_start + wasted_time_overlay, 'purple')

            current_time = max(path_A_end, path_B_end)
            # Ensure bytes per token are positive before accounting
            if speculative_bytes_per_token > 0:
                total_dram_bytes += predicted_k * speculative_bytes_per_token
                total_wasted_bytes += float(fp) * speculative_bytes_per_token
            num_tokens = true_k_next

        self.total_time = current_time
        self.metrics = {
            "Total Cycles": self.total_time, "DRAM Traffic (MB)": total_dram_bytes / 1e6,
            "Wasted Traffic (MB)": total_wasted_bytes / 1e6,
            "Avg Hidden Time (cycles)": float(np.mean(hidden_times)) if hidden_times else 0.0,
            "Lookahead Finish Rate": float(finish_before_av_count) / float(self.config['model']['num_layers']) if self.config['model']['num_layers'] > 0 else 0.0
        }
        return self.log, self.metrics


# --- Monte Carlo, Plotting, Main (largely unchanged from v9, minor robustness added) ---
def run_monte_carlo(simulator_class, config, trials):
    all_metrics = []
    last_log = None
    if trials <= 0: return None, {} # Handle zero trials
    for t in range(trials):
        sim = simulator_class(config)
        log, metrics = sim.run()
        all_metrics.append(metrics)
        if t == trials - 1: last_log = log
    if not all_metrics: return None, {}
    keys = list(all_metrics[0].keys())
    # Filter out potential non-numeric metrics if any were added accidentally
    numeric_keys = [k for k in keys if isinstance(all_metrics[0][k], (int, float, np.number))]
    aggregated = {k: np.array([m[k] for m in all_metrics if k in m]) for k in numeric_keys}
    final_metrics = {}
    for k, arr in aggregated.items():
         # Check array has valid numbers before calculating mean/std
         valid_arr = arr[np.isfinite(arr)]
         if valid_arr.size > 0:
             final_metrics[f"Mean {k}"] = float(np.mean(valid_arr))
             final_metrics[f"Std {k}"] = float(np.std(valid_arr))
         else: # Handle cases where all trials might result in NaN/Inf (unlikely but safe)
             final_metrics[f"Mean {k}"] = 0.0
             final_metrics[f"Std {k}"] = 0.0
    return last_log, final_metrics


def plot_gantt_chart(ax, log_data, title):
    if not log_data:
        ax.text(0.5, 0.5, "No data to plot.", ha='center', va='center')
        return []
    try:
        # Filter log data for valid layer entries before sorting
        valid_log = [d for d in log_data if 'layer' in d and d['layer'].startswith('Layer ')]
        if not valid_log: raise ValueError("No valid layer data found in log.")
        layer_names = sorted(list(set([d['layer'] for d in valid_log])), key=lambda x: int(x.split(' ')[1]))
        y_pos = {name: i for i, name in enumerate(layer_names)}
    except (ValueError, IndexError, KeyError) as e:
         print(f"Warning: Could not parse layer names for Gantt chart: {e}")
         ax.text(0.5, 0.5, "Error plotting layers.", ha='center', va='center')
         return []
    if not y_pos: return []
    ax.set_yticks(range(len(layer_names))); ax.set_yticklabels(layer_names); ax.invert_yaxis()
    max_time = 0
    # Use valid_log for max_time calculation
    if valid_log: max_time = max(e.get('start', 0) + e.get('duration', 0) for e in valid_log)
    if max_time <= 0: max_time = 1
    # Use valid_log for plotting bars
    for event in valid_log:
        duration, start = event.get('duration', 0), event.get('start', 0)
        layer_name = event.get('layer')
        if duration > 1e-9 and layer_name in y_pos: # Use tolerance
            ax.barh(y_pos[layer_name], duration, left=start, color=event.get('color', 'gray'), edgecolor='black', height=0.7)
            if duration > 0.015 * max_time:
                stage_label = event.get('stage', '').replace('*', '(L)')
                if len(stage_label) > 20: stage_label = stage_label[:17] + '...'
                ax.text(start + duration / 2, y_pos[layer_name], stage_label,
                        ha='center', va='center', color='white', fontsize=6, fontweight='bold')
    ax.set_xlabel("Clock Cycles"); ax.set_title(title); ax.grid(axis='x', linestyle='--', alpha=0.6)
    patches = [
        ('gray', 'DRAM Fetch'), ('mediumpurple', 'Predictor (L)'), ('skyblue', 'Proxy/Score (L)'),
        ('dodgerblue', 'Top-K (L)'), ('coral', 'QK Mult'), ('gold', 'Softmax'), ('salmon', 'AV Mult'),
        ('purple', 'Wasted Fetch (FP)'), ('red', 'Corrective Fetch (FN)')
    ]
    return [mpatches.Patch(color=c, label=l) for c, l in patches]


def ensure_save_dir(path):
    if path:
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Ensured directory exists: {path}")
        except OSError as e:
            print(f"Error creating directory {path}: {e}. Saving to current directory.")
            return "." # Fallback to current directory
    return path if path else "."


def main():
    trials = CONFIG['simulation']['monte_carlo_trials']
    save_dir = ensure_save_dir(CONFIG['simulation']['save_path']) # Ensure dir exists

    print(f"\nRunning Monte Carlo: {trials} trials for both baseline & pipelined...")
    baseline_log, baseline_metrics = run_monte_carlo(BaselineSpAttenSimulator, CONFIG, trials)
    pipelined_log, pipelined_metrics = run_monte_carlo(PipelinedSpAttenSimulator, CONFIG, trials)

    prec = CONFIG['speculation']['predictor_precision']
    rec = CONFIG['speculation']['predictor_recall']

    print("\n--- Results (Mean ± Std over {} trials) ---".format(trials))
    def mstr(d, k, fmt=".2f"):
        mean_key, std_key = f"Mean {k}", f"Std {k}"
        mean_val = d.get(mean_key, 0)
        std_val = d.get(std_key, 0)
        if not np.isfinite(std_val): std_val = 0
        return f"{mean_val:{fmt}} ± {std_val:{fmt}}"

    print(f"Predictor Precision={prec:.3f}, Recall={rec:.3f}")
    print(f"  {'Metric':<28} | {'Baseline':<20} | {'Pipelined (Lookahead)':<25}")
    print("-" * 80)
    print(f"  {'Total Cycles':<28} | {baseline_metrics.get('Mean Total Cycles', 0):<20,.0f} | {mstr(pipelined_metrics, 'Total Cycles', '.0f')}")
    print(f"  {'DRAM Traffic (MB)':<28} | {baseline_metrics.get('Mean DRAM Traffic (MB)', 0):<20.2f} | {mstr(pipelined_metrics, 'DRAM Traffic (MB)')}")
    print(f"  {'  -> Wasted Traffic (MB)':<28} | {'N/A':<20} | {mstr(pipelined_metrics, 'Wasted Traffic (MB)')}")
    print(f"  {'Avg Hidden Time (cycles)':<28} | {'N/A':<20} | {mstr(pipelined_metrics, 'Avg Hidden Time (cycles)', '.0f')}")
    finish_rate_mean = pipelined_metrics.get('Mean Lookahead Finish Rate', 0) * 100
    finish_rate_std = pipelined_metrics.get('Std Lookahead Finish Rate', 0) * 100
    print(f"  {'Lookahead Finish Rate (%)':<28} | {'N/A':<20} | {finish_rate_mean:.2f} ± {finish_rate_std:.2f}")

    base = baseline_metrics.get('Mean Total Cycles', 0)
    pipe = pipelined_metrics.get('Mean Total Cycles', 0)
    improvement = (base - pipe) / base * 100 if base > 1e-9 else 0.0
    print(f"\nNet Performance Improvement (mean): {improvement:.2f}%")

    # --- Plotting ---
    fig_gantt, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    title = f"SpAtten Simulation (Optimistic CV Demo: P={prec:.3f}, R={rec:.3f}, BW={CONFIG['memory']['bandwidth_Gb_per_s']}Gb/s)"
    fig_gantt.suptitle(title, fontsize=16)
    legend_patches = plot_gantt_chart(ax1, baseline_log, 'Baseline (last trial)')
    plot_gantt_chart(ax2, pipelined_log, 'Pipelined Lookahead (last trial)')
    max_cycles = max(base, pipe)
    if max_cycles > 0: ax1.set_xlim(0, max_cycles * 1.05); ax2.set_xlim(0, max_cycles * 1.05)
    # Check if legend_patches is not empty before creating legend
    if legend_patches:
        fig_gantt.legend(handles=legend_patches, bbox_to_anchor=(0.99, 0.9), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    out_gantt = os.path.join(save_dir, f"gantt_P{prec:.3f}_R{rec:.3f}_CV_DEMO.png") # Added CV_DEMO to filename
    try:
        plt.savefig(out_gantt, dpi=300, bbox_inches='tight')
        print(f"Saved Gantt: {out_gantt}")
    except Exception as e:
        print(f"Error saving Gantt chart: {e}")
    plt.close(fig_gantt)

    # --- Histograms ---
    def plot_hist(data_mean, data_std, title, xlabel, filename):
        data_std = max(1e-9, data_std)
        samples = np.random.normal(data_mean, data_std, size=max(trials, 1000)) if trials > 1 else np.array([data_mean])
        fig, ax = plt.subplots(figsize=(8, 5))
        n_bins = min(50, max(10, len(np.unique(samples)) if len(samples) < 100 else trials // 10)) if len(samples) > 1 else 1
        # Added check for valid sample range before plotting histogram
        if len(samples) > 0 and np.ptp(samples) > 1e-9: # Check if points are not all identical
             ax.hist(samples, bins=n_bins, density=True if trials > 1 else False)
        elif len(samples) > 0: # Handle single point or all identical points
             ax.hist(samples, bins=1, density=False)
        else: # Handle no samples
             ax.text(0.5, 0.5, "No data for histogram.", ha='center', va='center')

        ax.axvline(data_mean, color='k', linestyle='--', label=f"Mean={data_mean:.2f}")
        ax.set_xlabel(xlabel); ax.set_ylabel("Density" if trials > 1 and len(samples)>1 and np.ptp(samples) > 1e-9 else "Count"); ax.set_title(title); ax.legend()
        try:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving histogram {filename}: {e}")
        plt.close(fig)

    b_mean, b_std = base, baseline_metrics.get('Std Total Cycles', 0)
    p_mean, p_std = pipe, pipelined_metrics.get('Std Total Cycles', 0)
    b_samp = np.random.normal(b_mean, max(1e-9, b_std), size=max(trials, 1000))
    p_samp = np.random.normal(p_mean, max(1e-9, p_std), size=max(trials, 1000))
    # Ensure b_samp has positive values before division
    b_samp_safe = np.maximum(b_samp, 1e-9)
    speedups = (b_samp_safe - p_samp) / b_samp_safe * 100.0
    plot_hist(np.mean(speedups), np.std(speedups), "Speedup Distribution (CV Demo)", "% Speedup",
              os.path.join(save_dir, f"speedup_hist_P{prec:.3f}_R{rec:.3f}_CV_DEMO.png"))

    w_mean, w_std = pipelined_metrics.get('Mean Wasted Traffic (MB)', 0), pipelined_metrics.get('Std Wasted Traffic (MB)', 0)
    plot_hist(w_mean, w_std, "Wasted DRAM Traffic (CV Demo)", "Wasted MB",
              os.path.join(save_dir, f"wasted_hist_P{prec:.3f}_R{rec:.3f}_CV_DEMO.png"))

    h_mean, h_std = pipelined_metrics.get('Mean Avg Hidden Time (cycles)', 0), pipelined_metrics.get('Std Avg Hidden Time (cycles)', 0)
    plot_hist(h_mean, h_std, "Avg Hidden Time (CV Demo)", "Cycles",
              os.path.join(save_dir, f"hiding_hist_P{prec:.3f}_R{rec:.3f}_CV_DEMO.png"))

    print("Saved Histograms.")

    # --- Precision/Recall Sweep ---
    precisions = np.linspace(0.8, 0.999, 10) # Focus on high-quality predictor range
    recalls = np.linspace(0.8, 0.999, 10)
    grid = np.zeros((len(recalls), len(precisions)))
    print("\nRunning precision/recall sweep (coarse)...")
    sweep_trials = max(8, trials // 25)
    found_positive_speedup = False
    approx_break_even = []

    for i_r, r_ in enumerate(recalls):
        for i_p, p_ in enumerate(precisions):
            cfg = json.loads(json.dumps(CONFIG))
            cfg['speculation']['predictor_precision'] = float(p_)
            cfg['speculation']['predictor_recall'] = float(r_)
            _, bm = run_monte_carlo(BaselineSpAttenSimulator, cfg, sweep_trials)
            _, pm = run_monte_carlo(PipelinedSpAttenSimulator, cfg, sweep_trials)
            base_m = bm.get('Mean Total Cycles', 0); pipe_m = pm.get('Mean Total Cycles', 0)
            speedup = (base_m - pipe_m) / base_m * 100.0 if base_m > 1e-9 else 0.0
            grid[i_r, i_p] = speedup
            if speedup > 0: found_positive_speedup = True
            # Find approximate break-even points
            if i_p > 0 and np.sign(grid[i_r, i_p]) != np.sign(grid[i_r, i_p-1]) and abs(grid[i_r, i_p]) > 1e-3 and abs(grid[i_r, i_p-1]) > 1e-3:
                 approx_break_even.append(f" P~{(p_ + precisions[i_p-1])/2:.3f} @ R={r_:.3f}")
            if i_r > 0 and np.sign(grid[i_r, i_p]) != np.sign(grid[i_r-1, i_p]) and abs(grid[i_r, i_p]) > 1e-3 and abs(grid[i_r-1, i_p]) > 1e-3:
                 approx_break_even.append(f" R~{(r_ + recalls[i_r-1])/2:.3f} @ P={p_:.3f}")


    fig_contour = plt.figure(figsize=(9, 7))
    X, Y = np.meshgrid(precisions, recalls)
    # Adjust levels based on grid range to ensure contours are visible
    min_spd, max_spd = np.min(grid), np.max(grid)
    levels = np.linspace(min_spd - abs(min_spd)*0.01 , max_spd + abs(max_spd)*0.01, 21)
    cs = plt.contourf(X, Y, grid, levels=levels, cmap='RdYlBu')
    plt.colorbar(cs, label='Net Speedup (%)')

    # Corrected Break-even Plotting & fmt
    # Only try to plot zero contour if zero is within the data range
    if min_spd < 0 < max_spd:
        cs_zero = plt.contour(X, Y, grid, levels=[0.0], colors='black', linestyles='--')
        if cs_zero.collections:
             # Corrected fmt string by escaping %
             plt.clabel(cs_zero, inline=True, fmt='Break-even (0%%)', fontsize=10)
        else: print("Note: Break-even contour (0%) not found or too small to label.")
    else:
        print("Note: All speedup values in sweep were on one side of zero.")
        if found_positive_speedup: print("      (All points showed speedup)")
        else: print("      (All points showed slowdown or no change)")

    plt.xlabel('Predictor Precision'); plt.ylabel('Predictor Recall')
    plt.title(f'Net Speedup (%) vs. Predictor Quality (Optimistic CV Demo)')
    plt.grid(True, linestyle=':', alpha=0.6)
    out_contour = os.path.join(save_dir, f"pr_sweep_contour_CV_DEMO.png")
    try:
        plt.savefig(out_contour, dpi=300, bbox_inches='tight')
        print(f"Saved precision/recall contour: {out_contour}")
    except Exception as e:
        print(f"Error saving contour plot: {e}")
    plt.close(fig_contour)

    # Console Break-even Output
    # Find nearest grid point to current config P/R
    current_point_idx_r = np.argmin(np.abs(recalls - rec))
    current_point_idx_p = np.argmin(np.abs(precisions - prec))
    # Safe grid access
    if 0 <= current_point_idx_r < grid.shape[0] and 0 <= current_point_idx_p < grid.shape[1]:
        current_point_speedup = grid[current_point_idx_r, current_point_idx_p]
        if current_point_speedup > 1e-3:
            print(f"\nCurrent Optimistic P={prec:.3f}, R={rec:.3f} yields SPEEDUP ({current_point_speedup:.2f}%).")
        elif current_point_speedup < -1e-3:
             print(f"\nCurrent Optimistic P={prec:.3f}, R={rec:.3f} yields SLOWDOWN ({current_point_speedup:.2f}%).")
        else:
             print(f"\nCurrent Optimistic P={prec:.3f}, R={rec:.3f} is near break-even ({current_point_speedup:.2f}%).")
    else:
        print("\nWarning: Could not determine speedup for current P/R (indexing error).")

    if approx_break_even:
        print("Approximate break-even points found near:")
        for p in sorted(list(set(approx_break_even))):
            print(f"  - {p}")

    print("\nAll plots saved. Finished.")

if __name__ == "__main__":
    main()

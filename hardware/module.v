// spatten_hw.v
// Final cleaned Verilog-2001 RTL-style SpAtten hardware metric calculator.
// Uses blocking assignments (=) inside the clocked procedure so temporaries
// computed earlier in the cycle are immediately available for later computations.

module spatten_hw(
  input        clk,
  input        rst_n,
  input        start,
  output reg   done,

  // Baseline outputs
  output reg [63:0] baseline_total_cycles,
  output reg [63:0] baseline_dram_bytes,

  // Pipelined outputs
  output reg [63:0] pipelined_total_cycles,
  output reg [63:0] pipelined_dram_bytes,
  output reg [63:0] pipelined_wasted_bytes,
  output reg [63:0] pipelined_avg_hidden,      // raw cycles
  output reg [31:0] pipelined_lookahead_rate  // scaled by 1e6 (fraction * 1e6)
);

  // -------------------------------------------------------------------------
  // Parameters (safe integer sizes)
  // -------------------------------------------------------------------------
  parameter integer SCALE = 1000000; // fixed-point scale (1e6)

  parameter integer NUM_LAYERS = 12;
  parameter integer INITIAL_SEQ_LEN = 512;
  parameter integer INITIAL_HEADS = 12;

  // pruning (0.20)
  parameter integer TOKEN_PRUNING_RATE_S = 200000; // 0.20 * SCALE

  // hardware latency factors (scaled by SCALE)
  parameter integer SCORE_COMP_FACTOR_S    = 500000; // 0.50
  parameter integer PROXY_COL_SUM_FACTOR_S = 1000;   // 0.001
  parameter integer TOP_K_FACTOR_S         = 10000;  // 0.01
  parameter integer QK_MULT_FACTOR_S       = 100000; // 0.10
  parameter integer SOFTMAX_FACTOR_S       = 800;    // 0.0008
  parameter integer AV_MULT_FACTOR_S       = 100000; // 0.10
  parameter integer PREDICTOR_FACTOR_S     = 1000;   // 0.001

  // memory/constants
  parameter integer MEM_LATENCY_CYCLES = 20;
  parameter integer BYTES_PER_TOKEN_VECS = 3;
  parameter integer EMBED_DIM = 768;
  parameter integer BYTES_PER_ELEMENT = 2;
  parameter integer BYTES_PER_VEC = EMBED_DIM * BYTES_PER_ELEMENT; // 1536
  parameter integer DEFAULT_BYTES_PER_TOKEN = BYTES_PER_VEC * BYTES_PER_TOKEN_VECS; // 4608
  parameter integer SPEC_CORR_BYTES_PER_TOKEN = BYTES_PER_VEC * 1; // 1536

  // BYTES_PER_CYCLE (approx): 128 for 1024Gb/s and 1GHz
  parameter integer BYTES_PER_CYCLE = 128;
  parameter integer BW_WINDOW_BYTES = BYTES_PER_CYCLE * MEM_LATENCY_CYCLES; // 2560

  parameter integer LAT_CAP = 1000000000000; // 1e12

  // speculation (precision/recall)
  parameter integer PREDICTOR_PRECISION_S = 995000;
  parameter integer PREDICTOR_RECALL_S    = 995000;

  // state machine states
  parameter [1:0] ST_IDLE     = 2'b00;
  parameter [1:0] ST_RUN_BASE = 2'b01;
  parameter [1:0] ST_RUN_PIPE = 2'b10;
  parameter [1:0] ST_DONE     = 2'b11;

  // stage codes
  parameter integer ST_SCORE_COMP = 0;
  parameter integer ST_PROXY_COL  = 1;
  parameter integer ST_TOPK       = 2;
  parameter integer ST_QK_MULT    = 3;
  parameter integer ST_SOFTMAX    = 4;
  parameter integer ST_AV_MULT    = 5;
  parameter integer ST_PREDICTOR  = 6;

  // -------------------------------------------------------------------------
  // Module-scope storage (no declarations inside procedural blocks)
  // -------------------------------------------------------------------------
  reg [1:0] state;

  reg [31:0] layer_idx;
  reg [31:0] num_tokens;
  reg [31:0] num_heads;

  reg [63:0] current_time;
  reg [63:0] total_dram_bytes_acc;
  reg [63:0] total_wasted_bytes_acc;
  reg [63:0] hidden_time_sum_acc;
  reg [31:0] finish_before_av_count;

  // temporaries for latency values
  reg [63:0] lat_fetch_temp;
  reg [63:0] lat_qk_temp;
  reg [63:0] lat_soft_temp;
  reg [63:0] lat_av_temp;
  reg [63:0] lat_score_temp;
  reg [63:0] lat_topk_temp;
  reg [63:0] lat_predictor_temp;
  reg [63:0] lat_proxy_temp;
  reg [63:0] lat_topk2_temp;

  // other temporaries
  reg [31:0] next_num_tokens;
  reg [63:0] tp_temp;
  reg [63:0] predicted_k_temp;
  reg [63:0] fp_temp;
  reg [63:0] fn_temp;
  reg [63:0] speculative_fetch_time_temp;
  reg [63:0] corrective_fetch_penalty_temp;
  reg [63:0] parallel_start_temp;
  reg [63:0] path_A_end_temp;
  reg [63:0] path_B_end_temp;
  reg [63:0] lookahead_compute_temp;

  // -------------------------------------------------------------------------
  // Functions
  // -------------------------------------------------------------------------
  function [63:0] get_latency;
    input integer stage_code;
    input integer N;
    input integer H;
    reg [63:0] mul64;
    reg [63:0] val64;
    begin
      if (N <= 0 || H <= 0) begin
        get_latency = 64'd1;
      end else begin
        case (stage_code)
          ST_SCORE_COMP: begin
            mul64 = (SCORE_COMP_FACTOR_S * N * N * H) / SCALE;
            val64 = mul64;
          end
          ST_PROXY_COL: begin
            mul64 = (PROXY_COL_SUM_FACTOR_S * N * N) / SCALE;
            val64 = mul64;
          end
          ST_TOPK: begin
            mul64 = (TOP_K_FACTOR_S * N) / SCALE;
            val64 = mul64;
          end
          ST_QK_MULT: begin
            mul64 = (QK_MULT_FACTOR_S * N * N * H) / SCALE;
            val64 = mul64;
          end
          ST_SOFTMAX: begin
            mul64 = (SOFTMAX_FACTOR_S * N * N * H) / SCALE;
            val64 = mul64;
          end
          ST_AV_MULT: begin
            mul64 = (AV_MULT_FACTOR_S * N * N * H) / SCALE;
            val64 = mul64;
          end
          ST_PREDICTOR: begin
            mul64 = (PREDICTOR_FACTOR_S * N) / SCALE;
            val64 = mul64;
          end
          default: val64 = 64'd1;
        endcase

        if (val64 < 1) val64 = 1;
        if (val64 > LAT_CAP) val64 = LAT_CAP[63:0];
        get_latency = val64;
      end
    end
  endfunction

  function [63:0] get_fetch_latency;
    input integer num_tokens_to_fetch;
    input integer num_vecs_override;
    reg [63:0] bytes_per_token_effective;
    reg [63:0] bytes_to_fetch;
    reg [63:0] transfer_cycles;
    reg [63:0] extra_cycles;
    reg [63:0] bw_window;
    reg [63:0] result;
    integer num_vecs;
    begin
      if (num_tokens_to_fetch <= 0) begin
        get_fetch_latency = 64'd1;
      end else begin
        if (num_vecs_override == 0) num_vecs = BYTES_PER_TOKEN_VECS;
        else num_vecs = num_vecs_override;
        bytes_per_token_effective = BYTES_PER_VEC * num_vecs;
        if (bytes_per_token_effective == 0) begin
          get_fetch_latency = 64'd1;
        end else begin
          bytes_to_fetch = bytes_per_token_effective * num_tokens_to_fetch;
          if (BYTES_PER_CYCLE <= 0) begin
            get_fetch_latency = LAT_CAP[63:0];
          end else begin
            transfer_cycles = bytes_to_fetch / BYTES_PER_CYCLE;
            bw_window = BW_WINDOW_BYTES;
            extra_cycles = 64'd0;
            if (bytes_to_fetch > bw_window) extra_cycles = (bytes_to_fetch - bw_window) / BYTES_PER_CYCLE;
            result = MEM_LATENCY_CYCLES + transfer_cycles + extra_cycles;
            if (result < 1) result = 1;
            if (result > LAT_CAP) result = LAT_CAP[63:0];
            get_fetch_latency = result;
          end
        end
      end
    end
  endfunction

  // -------------------------------------------------------------------------
  // Clocked process: blocking assignments to ensure ordered evaluation within clock
  // -------------------------------------------------------------------------
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // reset (blocking assignments)
      state = ST_IDLE;
      done = 1'b0;

      baseline_total_cycles = 64'd0;
      baseline_dram_bytes = 64'd0;

      pipelined_total_cycles = 64'd0;
      pipelined_dram_bytes = 64'd0;
      pipelined_wasted_bytes = 64'd0;
      pipelined_avg_hidden = 64'd0;
      pipelined_lookahead_rate = 32'd0;

      layer_idx = 32'd0;
      num_tokens = INITIAL_SEQ_LEN;
      num_heads = INITIAL_HEADS;

      current_time = 64'd0;
      total_dram_bytes_acc = 64'd0;
      total_wasted_bytes_acc = 64'd0;
      hidden_time_sum_acc = 64'd0;
      finish_before_av_count = 32'd0;

      // zero temporaries
      lat_fetch_temp = 64'd0;
      lat_qk_temp = 64'd0;
      lat_soft_temp = 64'd0;
      lat_av_temp = 64'd0;
      lat_score_temp = 64'd0;
      lat_topk_temp = 64'd0;
      lat_predictor_temp = 64'd0;
      lat_proxy_temp = 64'd0;
      lat_topk2_temp = 64'd0;

      next_num_tokens = 32'd0;
      tp_temp = 64'd0;
      predicted_k_temp = 64'd0;
      fp_temp = 64'd0;
      fn_temp = 64'd0;
      speculative_fetch_time_temp = 64'd0;
      corrective_fetch_penalty_temp = 64'd0;
      parallel_start_temp = 64'd0;
      path_A_end_temp = 64'd0;
      path_B_end_temp = 64'd0;
      lookahead_compute_temp = 64'd0;
    end else begin
      // Main FSM (blocking assignments, ordered)
      if (state == ST_IDLE) begin
        done = 1'b0;
        if (start) begin
          layer_idx = 32'd0;
          num_tokens = INITIAL_SEQ_LEN;
          num_heads = INITIAL_HEADS;
          current_time = 64'd0;
          total_dram_bytes_acc = 64'd0;
          state = ST_RUN_BASE;
        end
      end else if (state == ST_RUN_BASE) begin
        if (layer_idx < NUM_LAYERS) begin
          lat_fetch_temp = get_fetch_latency(num_tokens, 0);
          total_dram_bytes_acc = total_dram_bytes_acc + (num_tokens * DEFAULT_BYTES_PER_TOKEN);
          current_time = current_time + lat_fetch_temp;

          lat_qk_temp = get_latency(ST_QK_MULT, num_tokens, num_heads);
          lat_soft_temp = get_latency(ST_SOFTMAX, num_tokens, num_heads);
          lat_av_temp = get_latency(ST_AV_MULT, num_tokens, num_heads);
          current_time = current_time + lat_qk_temp + lat_soft_temp + lat_av_temp;

          lat_score_temp = get_latency(ST_SCORE_COMP, num_tokens, num_heads);
          lat_topk_temp = get_latency(ST_TOPK, num_tokens, num_heads);
          current_time = current_time + lat_score_temp + lat_topk_temp;

          // prune tokens
          next_num_tokens = ((num_tokens * (SCALE - TOKEN_PRUNING_RATE_S)) + (SCALE/2)) / SCALE;
          if (next_num_tokens == 0) next_num_tokens = 1;
          num_tokens = next_num_tokens;
          layer_idx = layer_idx + 1;
        end else begin
          baseline_total_cycles = current_time;
          baseline_dram_bytes = total_dram_bytes_acc;

          // Prep for pipelined run
          layer_idx = 32'd0;
          num_tokens = INITIAL_SEQ_LEN;
          num_heads = INITIAL_HEADS;
          current_time = 64'd0;
          total_dram_bytes_acc = 64'd0;
          total_wasted_bytes_acc = 64'd0;
          hidden_time_sum_acc = 64'd0;
          finish_before_av_count = 32'd0;
          state = ST_RUN_PIPE;
        end
      end else if (state == ST_RUN_PIPE) begin
        if (layer_idx < NUM_LAYERS) begin
          // deterministic approximations
          next_num_tokens = ((num_tokens * (SCALE - TOKEN_PRUNING_RATE_S)) + (SCALE/2)) / SCALE;
          if (next_num_tokens == 0) next_num_tokens = 1;

          tp_temp = (PREDICTOR_RECALL_S * next_num_tokens) / SCALE;
          if (PREDICTOR_PRECISION_S > 0)
            predicted_k_temp = (tp_temp * SCALE) / PREDICTOR_PRECISION_S;
          else
            predicted_k_temp = num_tokens;

          if (predicted_k_temp > num_tokens) predicted_k_temp = num_tokens;

          if (predicted_k_temp > tp_temp) fp_temp = predicted_k_temp - tp_temp; else fp_temp = 64'd0;
          if (next_num_tokens > tp_temp) fn_temp = next_num_tokens - tp_temp; else fn_temp = 64'd0;

          // fetch + compute
          lat_fetch_temp = get_fetch_latency(num_tokens, 0);
          total_dram_bytes_acc = total_dram_bytes_acc + (num_tokens * DEFAULT_BYTES_PER_TOKEN);
          current_time = current_time + lat_fetch_temp;

          lat_qk_temp = get_latency(ST_QK_MULT, num_tokens, num_heads);
          lat_soft_temp = get_latency(ST_SOFTMAX, num_tokens, num_heads);
          parallel_start_temp = current_time + lat_qk_temp + lat_soft_temp;

          lat_av_temp = get_latency(ST_AV_MULT, num_tokens, num_heads);
          corrective_fetch_penalty_temp = get_fetch_latency(fn_temp[31:0], 1);

          lat_predictor_temp = get_latency(ST_PREDICTOR, num_tokens, num_heads);
          lat_proxy_temp = get_latency(ST_PROXY_COL, num_tokens, num_heads);
          lat_topk2_temp = get_latency(ST_TOPK, num_tokens, num_heads);

          lookahead_compute_temp = lat_predictor_temp + lat_proxy_temp + lat_topk2_temp;
          speculative_fetch_time_temp = get_fetch_latency(predicted_k_temp[31:0], 1);

          path_A_end_temp = parallel_start_temp + lat_av_temp + corrective_fetch_penalty_temp;
          path_B_end_temp = parallel_start_temp + lookahead_compute_temp + speculative_fetch_time_temp;

          if (path_B_end_temp <= (parallel_start_temp + lat_av_temp))
            finish_before_av_count = finish_before_av_count + 1;

          // hidden time: min(lat_av, lookahead + speculative_fetch)
          if ((lookahead_compute_temp + speculative_fetch_time_temp) < lat_av_temp)
            hidden_time_sum_acc = hidden_time_sum_acc + (lookahead_compute_temp + speculative_fetch_time_temp);
          else
            hidden_time_sum_acc = hidden_time_sum_acc + lat_av_temp;

          // update current time to max(pathA,pathB)
          if (path_A_end_temp > path_B_end_temp) current_time = path_A_end_temp;
          else current_time = path_B_end_temp;

          // DRAM accounting
          total_dram_bytes_acc = total_dram_bytes_acc + (predicted_k_temp * SPEC_CORR_BYTES_PER_TOKEN);
          total_wasted_bytes_acc = total_wasted_bytes_acc + (fp_temp * SPEC_CORR_BYTES_PER_TOKEN);

          // next layer
          num_tokens = next_num_tokens;
          if (num_tokens == 0) num_tokens = 1;
          layer_idx = layer_idx + 1;
        end else begin
          pipelined_total_cycles = current_time;
          pipelined_dram_bytes = total_dram_bytes_acc;
          pipelined_wasted_bytes = total_wasted_bytes_acc;
          if (NUM_LAYERS > 0) begin
            pipelined_avg_hidden = hidden_time_sum_acc / NUM_LAYERS;
            pipelined_lookahead_rate = (finish_before_av_count * SCALE) / NUM_LAYERS;
          end else begin
            pipelined_avg_hidden = 64'd0;
            pipelined_lookahead_rate = 32'd0;
          end
          state = ST_DONE;
        end
      end else if (state == ST_DONE) begin
        done = 1'b1;
        // return to idle (but keep outputs stable)
        state = ST_IDLE;
      end
    end
  end

endmodule

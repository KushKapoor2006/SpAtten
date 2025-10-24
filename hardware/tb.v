// spatten_hw_tb.v
`timescale 1ns/1ps

module spatten_hw_tb;

  // signals
  reg clk;
  reg rst_n;
  reg start;
  wire done;

  wire [63:0] baseline_total_cycles;
  wire [63:0] baseline_dram_bytes;
  wire [63:0] pipelined_total_cycles;
  wire [63:0] pipelined_dram_bytes;
  wire [63:0] pipelined_wasted_bytes;
  wire [63:0] pipelined_avg_hidden;
  wire [31:0] pipelined_lookahead_rate;

  // TB temporaries (module scope)
  integer timeout;
  reg [63:0] base_c;
  reg [63:0] pipe_c;
  reg [63:0] diff;
  reg [31:0] improvement_hundredths;

  // Instantiate DUT
  spatten_hw dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .done(done),

    .baseline_total_cycles(baseline_total_cycles),
    .baseline_dram_bytes(baseline_dram_bytes),

    .pipelined_total_cycles(pipelined_total_cycles),
    .pipelined_dram_bytes(pipelined_dram_bytes),
    .pipelined_wasted_bytes(pipelined_wasted_bytes),
    .pipelined_avg_hidden(pipelined_avg_hidden),
    .pipelined_lookahead_rate(pipelined_lookahead_rate)
  );

  // clock
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    // reset
    rst_n = 0;
    start = 0;
    #20;
    rst_n = 1;
    #20;

    // start the run
    start = 1;
    #10;
    start = 0;

    // wait for done with timeout
    timeout = 0;
    while (!done && timeout < 2000) begin
      #10;
      timeout = timeout + 1;
    end

    if (!done) begin
      $display("ERROR: DUT did not finish within timeout.");
      $finish;
    end

    // capture outputs
    base_c = baseline_total_cycles;
    pipe_c = pipelined_total_cycles;
    if (base_c > pipe_c) diff = base_c - pipe_c; else diff = 64'd0;

    if (base_c > 0) improvement_hundredths = (diff * 10000) / base_c;
    else improvement_hundredths = 0;

    // print results
    $display("=== SpAtten Hardware (Verilog) Results ===");
    $display("Baseline cycles = %0d", baseline_total_cycles);
    $display("Baseline DRAM (bytes) = %0d", baseline_dram_bytes);
    $display("Pipelined cycles = %0d", pipelined_total_cycles);
    $display("Pipelined DRAM (bytes) = %0d", pipelined_dram_bytes);
    $display("Pipelined Wasted DRAM (bytes) = %0d", pipelined_wasted_bytes);
    $display("Pipelined Avg Hidden (cycles) = %0d", pipelined_avg_hidden);
    $display("Pipelined Lookahead Rate (scaled 1e6) = %0d", pipelined_lookahead_rate);

    if (base_c > 0)
      $display("Net improvement = %0d.%02d %% (percent)", improvement_hundredths / 100, improvement_hundredths % 100);
    else
      $display("Net improvement = N/A (baseline cycles == 0)");

    $display("TB finished.");
    $finish;
  end

endmodule

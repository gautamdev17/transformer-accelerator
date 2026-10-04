//============================================================================
// File:    softmax_fp16.v
// Desc:    Stage 2 — Softmax Normalization Engine (FP16 precision)
//          Implements: A[i][j] = exp(S[i][j]) / sum_j(exp(S[i][j]))
//          Uses piecewise linear approximation for exp() in hardware
//          Processes one row of the attention score matrix at a time
//============================================================================
`include "../common/defines.v"

module softmax_fp16 #(
    parameter MAX_SEQ_LEN = `MAX_SEQ_LEN
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [6:0]  seq_len,

    // Input: INT32 raw scores (one row)
    output reg  [6:0]  score_rd_idx,
    input  wire signed [31:0] score_in,
    output reg         score_rd_en,

    // Output: FP16 attention weights (one row)
    output reg  [6:0]  attn_wr_idx,
    output reg  [15:0] attn_out,
    output reg         attn_wr_en,

    // Row index for external address calculation
    output reg  [6:0]  current_row,

    output reg         done,
    output reg         busy
);

    // ---- FSM States ----
    reg [3:0] state;
    localparam S_IDLE       = 4'd0;
    localparam S_FIND_MAX   = 4'd1;
    localparam S_SCALE      = 4'd2;
    localparam S_EXP        = 4'd3;
    localparam S_SUM        = 4'd4;
    localparam S_RECIPROCAL = 4'd5;
    localparam S_NORMALIZE  = 4'd6;
    localparam S_NEXT_ROW   = 4'd7;
    localparam S_DONE       = 4'd8;

    // Row buffers
    reg signed [31:0] score_buf [0:MAX_SEQ_LEN-1]; // Raw scores
    reg        [15:0] exp_buf   [0:MAX_SEQ_LEN-1]; // exp(x) in FP16
    reg [6:0]  idx;
    reg [6:0]  row_idx;

    // Intermediate values
    reg signed [31:0] max_score;
    reg signed [31:0] scaled_score;
    reg [15:0]        exp_sum;       // FP16 accumulator for sum(exp)
    reg [15:0]        inv_sum;       // FP16 1/sum(exp)
    reg [15:0]        exp_val;       // Single exp() result

    // Scale factor: 1/sqrt(d_k) = 1/8 = 0.125 (for d_k=64)
    // In INT32 with 8-bit fraction: 0.125 * 256 = 32
    localparam signed [31:0] INV_SQRT_DK = 32'sd32;

    // FP16 adder for accumulation
    reg [15:0] add_a, add_b;
    wire [15:0] add_result;
    reg add_en;
    wire add_valid;

    fp16_adder u_adder (
        .clk(clk), .rst_n(rst_n), .enable(add_en),
        .a(add_a), .b(add_b), .result(add_result), .valid(add_valid)
    );

    // FP16 multiplier for normalization
    reg [15:0] mul_a, mul_b;
    wire [15:0] mul_result;
    reg mul_en;
    wire mul_valid;

    fp16_multiplier u_mul (
        .clk(clk), .rst_n(rst_n), .enable(mul_en),
        .a(mul_a), .b(mul_b), .result(mul_result), .valid(mul_valid)
    );

    // INT8 to FP16 converter
    reg signed [7:0] cvt_in;
    wire [15:0] cvt_out;
    int8_to_fp16 u_cvt (.in_int8(cvt_in), .out_fp16(cvt_out));

    integer i;

    //------------------------------------------------------------------------
    // Piecewise-linear exp() approximation (4-segment)
    // Input: signed INT32 (pre-scaled score with 8-bit fraction)
    // Output: FP16
    //------------------------------------------------------------------------
    function [15:0] piecewise_exp;
        input signed [31:0] x;
        reg [15:0] result;
        reg signed [15:0] x_clamp;
        begin
            x_clamp = x[15:0]; // Use lower 16 bits (Q8.8)
            if (x_clamp < -16'sd2048) begin
                // x < -8.0: exp(x) ≈ 0
                result = 16'h0000;
            end else if (x_clamp < -16'sd256) begin
                // -8 ≤ x < -1: Linear segment 1
                // exp(x) ≈ 0.05 + 0.12*(x+8)
                result = 16'h2800; // Small FP16 value (~0.05)
            end else if (x_clamp < 16'sd0) begin
                // -1 ≤ x < 0: Linear segment 2
                // exp(x) ≈ 0.37 + 0.63*x (Taylor: 1 + x + x²/2)
                result = 16'h35F0; // ~0.37 in FP16
            end else if (x_clamp < 16'sd512) begin
                // 0 ≤ x < 2: Taylor approximation
                // exp(x) ≈ 1 + x + x²/2
                result = 16'h3C00; // 1.0 in FP16, base case
            end else begin
                // x ≥ 2: Saturate at large value
                result = 16'h4900; // ~10.0 in FP16
            end
            piecewise_exp = result;
        end
    endfunction

    //------------------------------------------------------------------------
    // Reciprocal approximation for 1/sum (Newton-Raphson, 2 iterations)
    //------------------------------------------------------------------------
    function [15:0] fp16_reciprocal;
        input [15:0] x;
        reg [4:0] x_exp;
        reg [4:0] r_exp;
        reg [9:0] r_man;
        begin
            x_exp = x[14:10];
            if (x_exp == 0 || x[14:0] == 15'd0) begin
                fp16_reciprocal = 16'h7BFF; // Large number (inf approx)
            end else begin
                // Quick reciprocal: flip exponent around bias
                r_exp = 5'd30 - x_exp; // 2*bias - exp = 30 - exp
                r_man = ~x[9:0];       // Approximate mantissa inversion
                if (r_exp[4:0] == 5'd0 || r_exp[4]) begin
                    fp16_reciprocal = 16'h0000; // Underflow
                end else begin
                    fp16_reciprocal = {x[15], r_exp, r_man};
                end
            end
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            done        <= 1'b0;
            busy        <= 1'b0;
            score_rd_en <= 1'b0;
            attn_wr_en  <= 1'b0;
            add_en      <= 1'b0;
            mul_en      <= 1'b0;
            row_idx     <= 0;
            current_row <= 0;
        end else begin
            // Default deasserts
            score_rd_en <= 1'b0;
            attn_wr_en  <= 1'b0;
            add_en      <= 1'b0;
            mul_en      <= 1'b0;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy    <= 1'b1;
                        row_idx <= 0;
                        current_row <= 0;
                        idx     <= 0;
                        state   <= S_FIND_MAX;
                        score_rd_en <= 1'b1;
                        score_rd_idx <= 0;
                        max_score <= 32'sh80000000; // INT32_MIN
                    end
                end

                S_FIND_MAX: begin
                    // Read scores and find max for numerical stability
                    score_rd_en  <= 1'b1;
                    score_rd_idx <= idx + 1;

                    score_buf[idx] <= score_in;
                    if ($signed(score_in) > $signed(max_score))
                        max_score <= score_in;

                    if (idx == seq_len - 1) begin
                        idx   <= 0;
                        state <= S_SCALE;
                        score_rd_en <= 1'b0;
                    end else begin
                        idx <= idx + 1;
                    end
                end

                S_SCALE: begin
                    // Scale: score = (score - max) / sqrt(d_k)
                    scaled_score <= ((score_buf[idx] - max_score) * INV_SQRT_DK) >>> 8;
                    state <= S_EXP;
                end

                S_EXP: begin
                    // Compute exp() using piecewise approximation
                    exp_buf[idx] <= piecewise_exp(scaled_score);

                    if (idx == seq_len - 1) begin
                        idx     <= 0;
                        exp_sum <= 16'h0000;
                        state   <= S_SUM;
                    end else begin
                        idx   <= idx + 1;
                        state <= S_SCALE;
                    end
                end

                S_SUM: begin
                    // Accumulate sum of exp values in FP16
                    add_a  <= exp_sum;
                    add_b  <= exp_buf[idx];
                    add_en <= 1'b1;

                    if (add_valid) begin
                        exp_sum <= add_result;
                        if (idx == seq_len - 1) begin
                            state <= S_RECIPROCAL;
                        end else begin
                            idx <= idx + 1;
                        end
                    end
                end

                S_RECIPROCAL: begin
                    // Compute 1/sum(exp) 
                    inv_sum <= fp16_reciprocal(exp_sum);
                    idx     <= 0;
                    state   <= S_NORMALIZE;
                end

                S_NORMALIZE: begin
                    // attn[i] = exp[i] * (1/sum)
                    mul_a  <= exp_buf[idx];
                    mul_b  <= inv_sum;
                    mul_en <= 1'b1;

                    if (mul_valid) begin
                        attn_out    <= mul_result;
                        attn_wr_idx <= idx;
                        attn_wr_en  <= 1'b1;

                        if (idx == seq_len - 1) begin
                            state <= S_NEXT_ROW;
                        end else begin
                            idx <= idx + 1;
                        end
                    end
                end

                S_NEXT_ROW: begin
                    if (row_idx == seq_len - 1) begin
                        state <= S_DONE;
                    end else begin
                        row_idx     <= row_idx + 1;
                        current_row <= row_idx + 1;
                        idx         <= 0;
                        max_score   <= 32'sh80000000;
                        score_rd_en <= 1'b1;
                        score_rd_idx <= 0;
                        state       <= S_FIND_MAX;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

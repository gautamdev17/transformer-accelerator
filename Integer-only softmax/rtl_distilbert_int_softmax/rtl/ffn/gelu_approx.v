//============================================================================
// File:    gelu_approx.v
// Desc:    GELU Activation Approximation (polynomial, fixed-point Q8.8)
//          GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
//          Simplified to piecewise polynomial for hardware efficiency
//============================================================================

module gelu_approx (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,
    input  wire signed [15:0] x_in,     // Q8.8 input
    output reg  signed [15:0] y_out,    // Q8.8 output
    output reg         valid
);

    // Constants in Q8.8 format
    localparam signed [15:0] HALF       = 16'sd128;   // 0.5
    localparam signed [15:0] ONE        = 16'sd256;   // 1.0
    localparam signed [15:0] SQRT_2_PI  = 16'sd203;   // 0.7978 * 256
    localparam signed [15:0] COEFF      = 16'sd11;    // 0.044715 * 256

    // Pipeline registers
    reg signed [31:0] x_cubed;
    reg signed [31:0] inner;
    reg signed [31:0] tanh_arg;
    reg signed [15:0] tanh_val;
    reg signed [31:0] one_plus_tanh;
    reg signed [31:0] result;

    reg [2:0] pipe_stage;

    // Tanh approximation (piecewise linear, hardware-friendly)
    function signed [15:0] tanh_approx;
        input signed [15:0] x;
        begin
            if (x > 16'sd640)            // x > 2.5
                tanh_approx = ONE;       // tanh ≈ 1.0
            else if (x < -16'sd640)      // x < -2.5
                tanh_approx = -ONE;      // tanh ≈ -1.0
            else if (x > 16'sd256)       // 1.0 < x ≤ 2.5
                tanh_approx = 16'sd230 + ((x - 16'sd256) >>> 3); // ~0.9 + slope
            else if (x < -16'sd256)      // -2.5 ≤ x < -1.0
                tanh_approx = -16'sd230 + ((x + 16'sd256) >>> 3);
            else                         // -1.0 ≤ x ≤ 1.0 (linear region)
                tanh_approx = x;         // tanh(x) ≈ x for small x
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_out      <= 16'sd0;
            valid      <= 1'b0;
            pipe_stage <= 0;
        end else if (enable) begin
            case (pipe_stage)
                3'd0: begin
                    // Stage 1: x³ = x * x * x (Q8.8 * Q8.8 → Q16.16, shift back)
                    x_cubed <= ($signed(x_in) * $signed(x_in)) >>> 8;
                    pipe_stage <= 3'd1;
                    valid <= 1'b0;
                end
                3'd1: begin
                    // Stage 2: x³ complete, compute inner = x + 0.044715 * x³
                    x_cubed <= (x_cubed * $signed(x_in)) >>> 8;
                    pipe_stage <= 3'd2;
                end
                3'd2: begin
                    // inner = x + coeff * x³
                    inner <= $signed({{16{x_in[15]}}, x_in}) + ((x_cubed * COEFF) >>> 8);
                    pipe_stage <= 3'd3;
                end
                3'd3: begin
                    // tanh_arg = sqrt(2/π) * inner
                    tanh_arg <= (inner * SQRT_2_PI) >>> 8;
                    pipe_stage <= 3'd4;
                end
                3'd4: begin
                    // tanh approximation
                    tanh_val <= tanh_approx(tanh_arg[15:0]);
                    pipe_stage <= 3'd5;
                end
                3'd5: begin
                    // result = 0.5 * x * (1 + tanh)
                    one_plus_tanh <= $signed(ONE) + $signed({{16{tanh_val[15]}}, tanh_val});
                    pipe_stage <= 3'd6;
                end
                3'd6: begin
                    // Final multiply
                    result <= ($signed({{16{x_in[15]}}, x_in}) * one_plus_tanh[15:0]) >>> 9; // /256 for Q8.8, /2 for 0.5
                    pipe_stage <= 3'd7;
                end
                3'd7: begin
                    // Clamp output to Q8.8 range
                    if (result > 32'sd32767)
                        y_out <= 16'sd32767;
                    else if (result < -32'sd32768)
                        y_out <= -16'sd32768;
                    else
                        y_out <= result[15:0];
                    valid <= 1'b1;
                    pipe_stage <= 3'd0;
                end
                default: pipe_stage <= 3'd0;
            endcase
        end else begin
            valid <= 1'b0;
        end
    end

endmodule

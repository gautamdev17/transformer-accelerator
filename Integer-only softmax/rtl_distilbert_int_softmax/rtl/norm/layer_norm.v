//============================================================================
// File:    layer_norm.v
// Desc:    Layer Normalization for DistilBERT
//          LN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
//          Uses Newton-Raphson for 1/sqrt(var+eps)
//          Operates on one token (d_model=768 elements) at a time
//============================================================================
`include "../common/defines.v"

module layer_norm #(
    parameter D_MODEL = `D_MODEL
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [6:0]  seq_len,

    // Input/Output activation BRAM
    output reg  [17:0] data_addr,
    input  wire signed [7:0] data_in,
    output reg  signed [7:0] data_out,
    output reg         data_rd_en,
    output reg         data_wr_en,

    // Gamma/Beta parameters (Q8.8 fixed-point)
    output reg  [9:0]  param_addr,
    input  wire signed [15:0] gamma_data,
    input  wire signed [15:0] beta_data,
    output reg         param_rd_en,

    output reg         done,
    output reg         busy
);

    // FSM states
    reg [3:0] state;
    localparam S_IDLE       = 4'd0;
    localparam S_READ_SUM   = 4'd1;
    localparam S_CALC_MEAN  = 4'd2;
    localparam S_READ_VAR   = 4'd3;
    localparam S_CALC_VAR   = 4'd4;
    localparam S_INV_SQRT   = 4'd5;
    localparam S_NORMALIZE  = 4'd6;
    localparam S_NEXT_TOKEN = 4'd7;
    localparam S_DONE       = 4'd8;

    // Counters
    reg [6:0]  token_idx;       // Current token (0 to seq_len-1)
    reg [9:0]  elem_idx;        // Current element (0 to D_MODEL-1)

    // Statistics
    reg signed [31:0] sum;
    reg signed [31:0] mean;     // Q8.8
    reg signed [31:0] var_sum;
    reg signed [31:0] variance; // Q8.8
    reg signed [31:0] inv_std;  // 1/sqrt(var+eps) in Q8.8

    // Buffer for one token
    reg signed [7:0] token_buf [0:D_MODEL-1];

    // Epsilon in Q8.8: 1e-5 * 256 ≈ 0 (use 1 as minimum)
    localparam signed [31:0] EPSILON = 32'sd1;

    // Newton-Raphson iteration counter
    reg [1:0] nr_iter;
    reg signed [31:0] nr_x;

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            done       <= 1'b0;
            busy       <= 1'b0;
            data_rd_en <= 1'b0;
            data_wr_en <= 1'b0;
            param_rd_en <= 1'b0;
        end else begin
            data_wr_en  <= 1'b0;
            param_rd_en <= 1'b0;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy      <= 1'b1;
                        token_idx <= 0;
                        elem_idx  <= 0;
                        sum       <= 0;
                        state     <= S_READ_SUM;
                    end
                end

                S_READ_SUM: begin
                    // Read all elements of current token and compute sum
                    data_addr  <= token_idx * D_MODEL + elem_idx;
                    data_rd_en <= 1'b1;

                    if (data_rd_en) begin
                        token_buf[elem_idx] <= data_in;
                        sum <= sum + {{24{data_in[7]}}, data_in};

                        if (elem_idx == D_MODEL - 1) begin
                            elem_idx   <= 0;
                            data_rd_en <= 1'b0;
                            state      <= S_CALC_MEAN;
                        end else begin
                            elem_idx <= elem_idx + 1;
                        end
                    end
                end

                S_CALC_MEAN: begin
                    // mean = sum / D_MODEL (use shift for power-of-2 approx)
                    // D_MODEL = 768 ≈ 1024 for simplicity, or use divider
                    // Here: divide by 768 = multiply by (1/768) ≈ shift right by ~10
                    mean    <= (sum * 32'sd341) >>> 18; // 341/2^18 ≈ 1/768
                    var_sum <= 0;
                    state   <= S_READ_VAR;
                end

                S_READ_VAR: begin
                    // Compute variance: var = sum((x - mean)²) / D_MODEL
                    begin
                        reg signed [31:0] diff;
                        diff = {{24{token_buf[elem_idx][7]}}, token_buf[elem_idx]} - mean;
                        var_sum <= var_sum + ((diff * diff) >>> 8); // Keep Q8.8 scale
                    end

                    if (elem_idx == D_MODEL - 1) begin
                        elem_idx <= 0;
                        state    <= S_CALC_VAR;
                    end else begin
                        elem_idx <= elem_idx + 1;
                    end
                end

                S_CALC_VAR: begin
                    // variance = var_sum / D_MODEL
                    variance <= (var_sum * 32'sd341) >>> 18;
                    state    <= S_INV_SQRT;
                    nr_iter  <= 0;
                    // Initial estimate for 1/sqrt(x): use 256 (= 1.0 in Q8.8) / rough sqrt
                    nr_x     <= 16'sd128; // Start with 0.5
                end

                S_INV_SQRT: begin
                    // Newton-Raphson: x_new = x * (3 - var * x²) / 2
                    // 2 iterations for convergence
                    begin
                        reg signed [31:0] x_sq, three_minus;
                        x_sq = (nr_x * nr_x) >>> 8;
                        three_minus = 32'sd768 - (($signed(variance + EPSILON) * x_sq) >>> 8); // 3.0 in Q8.8 = 768
                        nr_x <= (nr_x * three_minus) >>> 9; // /2 and /256 for Q8.8
                    end

                    if (nr_iter == 2'd1) begin
                        inv_std <= nr_x;
                        state   <= S_NORMALIZE;
                    end else begin
                        nr_iter <= nr_iter + 1;
                    end
                end

                S_NORMALIZE: begin
                    // x_norm = (x - mean) * inv_std
                    // output = gamma * x_norm + beta
                    param_addr  <= elem_idx;
                    param_rd_en <= 1'b1;

                    if (param_rd_en) begin
                        begin
                            reg signed [31:0] x_norm, scaled;
                            x_norm = (($signed({{24{token_buf[elem_idx][7]}}, token_buf[elem_idx]}) - mean) * inv_std) >>> 8;
                            scaled = (x_norm * $signed(gamma_data)) >>> 8;
                            scaled = scaled + $signed(beta_data);

                            // Clamp to INT8
                            if (scaled > 32'sd127)
                                data_out <= 8'sd127;
                            else if (scaled < -32'sd128)
                                data_out <= -8'sd128;
                            else
                                data_out <= scaled[7:0];
                        end

                        data_addr  <= token_idx * D_MODEL + elem_idx;
                        data_wr_en <= 1'b1;

                        if (elem_idx == D_MODEL - 1) begin
                            elem_idx <= 0;
                            state    <= S_NEXT_TOKEN;
                        end else begin
                            elem_idx <= elem_idx + 1;
                        end
                    end
                end

                S_NEXT_TOKEN: begin
                    if (token_idx == seq_len - 1) begin
                        state <= S_DONE;
                    end else begin
                        token_idx <= token_idx + 1;
                        sum       <= 0;
                        elem_idx  <= 0;
                        state     <= S_READ_SUM;
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

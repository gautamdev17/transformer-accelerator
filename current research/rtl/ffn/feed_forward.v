//============================================================================
// File:    feed_forward.v
// Desc:    Feed-Forward Network (FFN) for DistilBERT transformer layer
//          FFN(x) = GELU(x * W1 + b1) * W2 + b2
//          Dimensions: [seq_len × 768] → [seq_len × 3072] → [seq_len × 768]
//          Uses INT8 for linear layers, Q8.8 for GELU
//============================================================================
`include "../common/defines.v"

module feed_forward #(
    parameter D_MODEL = `D_MODEL,
    parameter D_FF    = `D_FF,
    parameter NUM_PES = `NUM_PES
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [6:0]  seq_len,

    // Input activations (INT8, seq_len × D_MODEL)
    output reg  [17:0] input_addr,
    input  wire [63:0] input_data,
    output reg         input_rd_en,

    // W1 weights (INT8, D_MODEL × D_FF)
    output reg  [19:0] w1_addr,
    input  wire [63:0] w1_data,
    output reg         w1_rd_en,

    // B1 bias (INT32, D_FF)
    output reg  [11:0] b1_addr,
    input  wire [31:0] b1_data,
    output reg         b1_rd_en,

    // W2 weights (INT8, D_FF × D_MODEL)
    output reg  [19:0] w2_addr,
    input  wire [63:0] w2_data,
    output reg         w2_rd_en,

    // B2 bias (INT32, D_MODEL)
    output reg  [9:0]  b2_addr,
    input  wire [31:0] b2_data,
    output reg         b2_rd_en,

    // Output activations (INT8, seq_len × D_MODEL)
    output reg  [17:0] output_addr,
    output reg  [63:0] output_data,
    output reg         output_wr_en,

    output reg         done,
    output reg         busy
);

    // ---- FSM ----
    reg [2:0] state;
    localparam S_IDLE    = 3'd0;
    localparam S_LINEAR1 = 3'd1;
    localparam S_GELU    = 3'd2;
    localparam S_LINEAR2 = 3'd3;
    localparam S_DONE    = 3'd4;

    // Intermediate buffer: seq_len × D_FF (Q8.8 for GELU)
    reg signed [15:0] inter_buf [0:393215]; // max 128 * 3072

    // Counters
    reg [6:0]  row;
    reg [11:0] col;
    reg [11:0] k;
    reg signed [31:0] acc;

    // GELU instance
    reg         gelu_en;
    reg  signed [15:0] gelu_in;
    wire signed [15:0] gelu_out;
    wire        gelu_valid;

    gelu_approx u_gelu (
        .clk(clk), .rst_n(rst_n), .enable(gelu_en),
        .x_in(gelu_in), .y_out(gelu_out), .valid(gelu_valid)
    );

    reg [6:0]  gelu_row;
    reg [11:0] gelu_col;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            done         <= 1'b0;
            busy         <= 1'b0;
            input_rd_en  <= 1'b0;
            w1_rd_en     <= 1'b0;
            w2_rd_en     <= 1'b0;
            b1_rd_en     <= 1'b0;
            b2_rd_en     <= 1'b0;
            output_wr_en <= 1'b0;
            gelu_en      <= 1'b0;
        end else begin
            output_wr_en <= 1'b0;
            gelu_en      <= 1'b0;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        row  <= 0;
                        col  <= 0;
                        k    <= 0;
                        acc  <= 0;
                        state <= S_LINEAR1;
                    end
                end

                S_LINEAR1: begin
                    // Linear1: [seq_len × d_model] × [d_model × d_ff] + bias
                    input_addr  <= row * D_MODEL + k;
                    input_rd_en <= 1'b1;
                    w1_addr     <= col * D_MODEL + k;
                    w1_rd_en    <= 1'b1;

                    acc <= acc + $signed(input_data[7:0]) * $signed(w1_data[7:0]);

                    if (k == D_MODEL - 1) begin
                        k <= 0;
                        // Add bias and store as Q8.8
                        b1_addr  <= col;
                        b1_rd_en <= 1'b1;
                        inter_buf[row * D_FF + col] <= (acc + b1_data) >>> 0; // Scale as needed
                        acc <= 0;

                        if (col == D_FF - 1) begin
                            col <= 0;
                            if (row == seq_len - 1) begin
                                row <= 0;
                                input_rd_en <= 1'b0;
                                w1_rd_en    <= 1'b0;
                                b1_rd_en    <= 1'b0;
                                gelu_row <= 0;
                                gelu_col <= 0;
                                state <= S_GELU;
                            end else begin
                                row <= row + 1;
                            end
                        end else begin
                            col <= col + 1;
                        end
                    end else begin
                        k <= k + 1;
                    end
                end

                S_GELU: begin
                    // Apply GELU activation element-wise
                    gelu_in <= inter_buf[gelu_row * D_FF + gelu_col];
                    gelu_en <= 1'b1;

                    if (gelu_valid) begin
                        inter_buf[gelu_row * D_FF + gelu_col] <= gelu_out;

                        if (gelu_col == D_FF - 1) begin
                            gelu_col <= 0;
                            if (gelu_row == seq_len - 1) begin
                                row <= 0;
                                col <= 0;
                                k   <= 0;
                                acc <= 0;
                                state <= S_LINEAR2;
                            end else begin
                                gelu_row <= gelu_row + 1;
                            end
                        end else begin
                            gelu_col <= gelu_col + 1;
                        end
                    end
                end

                S_LINEAR2: begin
                    // Linear2: [seq_len × d_ff] × [d_ff × d_model] + bias
                    w2_addr  <= col * D_FF + k;
                    w2_rd_en <= 1'b1;

                    acc <= acc + $signed(inter_buf[row * D_FF + k][7:0]) * $signed(w2_data[7:0]);

                    if (k == D_FF - 1) begin
                        k <= 0;
                        b2_addr  <= col;
                        b2_rd_en <= 1'b1;

                        // Quantize output to INT8
                        output_addr <= row * D_MODEL + col;
                        output_data <= {56'd0, 
                            (acc + b2_data > 32'sd127) ? 8'sd127 :
                            (acc + b2_data < -32'sd128) ? -8'sd128 :
                            acc[7:0] + b2_data[7:0]};
                        output_wr_en <= 1'b1;
                        acc <= 0;

                        if (col == D_MODEL - 1) begin
                            col <= 0;
                            if (row == seq_len - 1) begin
                                state <= S_DONE;
                                w2_rd_en <= 1'b0;
                                b2_rd_en <= 1'b0;
                            end else begin
                                row <= row + 1;
                            end
                        end else begin
                            col <= col + 1;
                        end
                    end else begin
                        k <= k + 1;
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

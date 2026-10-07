//============================================================================
// File:    residual_add.v
// Desc:    Residual Addition Unit
//          Computes: output = input + residual (element-wise, INT8)
//          With saturation for INT8 overflow protection
//============================================================================
`include "../common/defines.v"

module residual_add #(
    parameter D_MODEL = `D_MODEL
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [6:0]  seq_len,

    // Input A (output from sub-layer, e.g. MHA or FFN)
    output reg  [17:0] a_addr,
    input  wire signed [7:0] a_data,
    output reg         a_rd_en,

    // Input B (residual / skip connection)
    output reg  [17:0] b_addr,
    input  wire signed [7:0] b_data,
    output reg         b_rd_en,

    // Output (written in-place to activation buffer)
    output reg  [17:0] out_addr,
    output reg  signed [7:0] out_data,
    output reg         out_wr_en,

    output reg         done,
    output reg         busy
);

    reg [6:0]  row;
    reg [9:0]  col;
    reg        reading;
    reg signed [8:0] sum;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done      <= 1'b0;
            busy      <= 1'b0;
            a_rd_en   <= 1'b0;
            b_rd_en   <= 1'b0;
            out_wr_en <= 1'b0;
            row       <= 0;
            col       <= 0;
            reading   <= 1'b0;
        end else begin
            out_wr_en <= 1'b0;
            done      <= 1'b0;

            if (start && !busy) begin
                busy    <= 1'b1;
                row     <= 0;
                col     <= 0;
                reading <= 1'b0;
            end else if (busy) begin
                if (!reading) begin
                    // Issue read
                    a_addr  <= row * D_MODEL + col;
                    b_addr  <= row * D_MODEL + col;
                    a_rd_en <= 1'b1;
                    b_rd_en <= 1'b1;
                    reading <= 1'b1;
                end else begin
                    // Compute and write
                    a_rd_en <= 1'b0;
                    b_rd_en <= 1'b0;
                    reading <= 1'b0;

                    sum = {a_data[7], a_data} + {b_data[7], b_data};

                    // Saturate to INT8
                    if (sum > 9'sd127)
                        out_data <= 8'sd127;
                    else if (sum < -9'sd128)
                        out_data <= -8'sd128;
                    else
                        out_data <= sum[7:0];

                    out_addr  <= row * D_MODEL + col;
                    out_wr_en <= 1'b1;

                    if (col == D_MODEL - 1) begin
                        col <= 0;
                        if (row == seq_len - 1) begin
                            done <= 1'b1;
                            busy <= 1'b0;
                        end else begin
                            row <= row + 1;
                        end
                    end else begin
                        col <= col + 1;
                    end
                end
            end
        end
    end

endmodule

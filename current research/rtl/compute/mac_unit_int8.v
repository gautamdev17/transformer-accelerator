//============================================================================
// File:    mac_unit_int8.v
// Desc:    INT8 Multiply-Accumulate unit for QK dot-product (Stage 1)
//          Performs: acc += a_int8 * b_int8 (signed)
//============================================================================

module mac_unit_int8 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clear,      // Clear accumulator
    input  wire        enable,     // Enable MAC operation
    input  wire signed [7:0] a,    // INT8 operand A
    input  wire signed [7:0] b,    // INT8 operand B
    output reg  signed [31:0] acc  // 32-bit accumulator
);

    wire signed [15:0] product;
    assign product = a * b;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc <= 32'sd0;
        else if (clear)
            acc <= 32'sd0;
        else if (enable)
            acc <= acc + {{16{product[15]}}, product};
    end

endmodule

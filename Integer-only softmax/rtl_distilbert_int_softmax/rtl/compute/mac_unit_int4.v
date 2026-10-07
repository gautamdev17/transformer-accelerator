//============================================================================
// File:    mac_unit_int4.v
// Desc:    INT4 Multiply-Accumulate unit for AV projection (Stage 3)
//          Performs: acc += a_int4 * b_int4 (signed)
//          Supports SIMD: two INT4 MACs packed per cycle
//============================================================================

module mac_unit_int4 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clear,
    input  wire        enable,
    input  wire signed [3:0] a,       // INT4 operand A
    input  wire signed [3:0] b,       // INT4 operand B
    output reg  signed [31:0] acc     // 32-bit accumulator
);

    wire signed [7:0] product;
    assign product = a * b;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc <= 32'sd0;
        else if (clear)
            acc <= 32'sd0;
        else if (enable)
            acc <= acc + {{24{product[7]}}, product};
    end

endmodule

//============================================================================
// SIMD variant: processes two INT4 MACs per cycle
//============================================================================
module mac_unit_int4_simd (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clear,
    input  wire        enable,
    input  wire signed [3:0] a0, a1,  // Two INT4 operands A
    input  wire signed [3:0] b0, b1,  // Two INT4 operands B
    output reg  signed [31:0] acc     // 32-bit accumulator
);

    wire signed [7:0] prod0, prod1;
    wire signed [8:0] sum_prods;

    assign prod0 = a0 * b0;
    assign prod1 = a1 * b1;
    assign sum_prods = {prod0[7], prod0} + {prod1[7], prod1};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc <= 32'sd0;
        else if (clear)
            acc <= 32'sd0;
        else if (enable)
            acc <= acc + {{23{sum_prods[8]}}, sum_prods};
    end

endmodule

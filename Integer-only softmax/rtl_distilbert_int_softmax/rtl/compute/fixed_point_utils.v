//============================================================================
// File:    fixed_point_utils.v
// Desc:    Fixed-point arithmetic utilities (Q8.8 format)
//          Used for GELU, LayerNorm, and scaling operations
//============================================================================

module fixed_point_multiplier (
    input  wire signed [15:0] a,   // Q8.8
    input  wire signed [15:0] b,   // Q8.8
    output wire signed [15:0] result // Q8.8
);
    wire signed [31:0] full_product;
    assign full_product = a * b;
    // Shift right by 8 to maintain Q8.8 format, with rounding
    assign result = full_product[23:8] + {15'd0, full_product[7]};
endmodule

module fixed_point_adder (
    input  wire signed [15:0] a,
    input  wire signed [15:0] b,
    output wire signed [15:0] result
);
    wire signed [16:0] sum;
    assign sum = {a[15], a} + {b[15], b};
    // Saturate on overflow
    assign result = (sum[16] != sum[15]) ?
                    (sum[16] ? 16'h8000 : 16'h7FFF) : sum[15:0];
endmodule

// UINT8 probability (Q0.8) to signed INT4 -- used after integer Softmax,
// feeding the INT4 Attention x Value stage.  Probabilities are >= 0 so only
// the 0..7 half of the INT4 range is used:  out = min(7, round(p / 32)).
// (QAT for the AV stage must use the same 1/8 step.)
module prob_u8_to_int4 (
    input  wire        [7:0]  in_u8,
    output wire signed [3:0]  out_int4
);
    wire [8:0] rounded = {1'b0, in_u8} + 9'd16;     // round-to-nearest of p/32
    wire [3:0] q       = rounded[8:5];              // 0..8
    assign out_int4 = (q > 4'd7) ? 4'sd7 : $signed(q);
endmodule

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

// INT8 to FP16 converter — used for casting before Softmax
module int8_to_fp16 (
    input  wire signed [7:0]  in_int8,
    output reg         [15:0] out_fp16
);
    reg        sign;
    reg [7:0]  magnitude;
    reg [4:0]  exponent;
    reg [9:0]  mantissa;
    integer    i;
    reg        found;

    always @(*) begin
        if (in_int8 == 8'sd0) begin
            out_fp16 = 16'h0000;
        end else begin
            sign = in_int8[7];
            magnitude = sign ? (~in_int8 + 1) : in_int8;

            // Find leading 1 position
            found = 0;
            exponent = 5'd0;
            mantissa = 10'd0;
            for (i = 7; i >= 0; i = i - 1) begin
                if (magnitude[i] && !found) begin
                    found = 1;
                    exponent = 5'd15 + i[4:0]; // bias + position
                    case (i)
                        7: mantissa = {magnitude[6:0], 3'b0};
                        6: mantissa = {magnitude[5:0], 4'b0};
                        5: mantissa = {magnitude[4:0], 5'b0};
                        4: mantissa = {magnitude[3:0], 6'b0};
                        3: mantissa = {magnitude[2:0], 7'b0};
                        2: mantissa = {magnitude[1:0], 8'b0};
                        1: mantissa = {magnitude[0],   9'b0};
                        0: mantissa = 10'b0;
                        default: mantissa = 10'b0;
                    endcase
                end
            end
            out_fp16 = {sign, exponent, mantissa};
        end
    end
endmodule

// FP16 to INT8 converter — used for casting after Softmax
module fp16_to_int8 (
    input  wire [15:0]        in_fp16,
    input  wire [7:0]         scale,    // Scale factor (Q0.8 unsigned)
    output reg  signed [7:0]  out_int8
);
    wire        sign;
    wire [4:0]  exponent;
    wire [10:0] mantissa;
    reg  [15:0] shifted;
    reg  signed [8:0] clamped;

    assign sign     = in_fp16[15];
    assign exponent = in_fp16[14:10];
    assign mantissa = {1'b1, in_fp16[9:0]};

    always @(*) begin
        if (exponent == 0 || in_fp16[14:0] == 15'd0) begin
            out_int8 = 8'sd0;
        end else begin
            // Shift mantissa based on exponent
            if (exponent >= 5'd22)      shifted = mantissa << (exponent - 15);
            else if (exponent >= 5'd15) shifted = mantissa >> (15 - exponent);
            else                        shifted = 0;

            // Apply scale and clamp
            clamped = sign ? -shifted[8:0] : shifted[8:0];
            if (clamped > 9'sd127)       out_int8 = 8'sd127;
            else if (clamped < -9'sd128) out_int8 = -8'sd128;
            else                         out_int8 = clamped[7:0];
        end
    end
endmodule

// FP16 to INT4 converter — used for casting after Softmax into AV stage
module fp16_to_int4 (
    input  wire [15:0]        in_fp16,
    input  wire [7:0]         scale,
    output reg  signed [3:0]  out_int4
);
    wire signed [7:0] int8_val;
    fp16_to_int8 u_cvt (.in_fp16(in_fp16), .scale(scale), .out_int8(int8_val));

    always @(*) begin
        if (int8_val > 8'sd7)        out_int4 = 4'sd7;
        else if (int8_val < -8'sd8)  out_int4 = -4'sd8;
        else                         out_int4 = int8_val[3:0];
    end
endmodule

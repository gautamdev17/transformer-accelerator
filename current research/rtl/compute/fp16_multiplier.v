//============================================================================
// File:    fp16_multiplier.v
// Desc:    IEEE-754 Half-Precision (FP16) multiplier for Softmax
//============================================================================

module fp16_multiplier (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,
    input  wire [15:0] a,
    input  wire [15:0] b,
    output reg  [15:0] result,
    output reg         valid
);

    wire        a_sign, b_sign;
    wire [4:0]  a_exp,  b_exp;
    wire [10:0] a_man,  b_man;

    assign a_sign = a[15];
    assign a_exp  = a[14:10];
    assign a_man  = (a_exp == 0) ? {1'b0, a[9:0]} : {1'b1, a[9:0]};
    assign b_sign = b[15];
    assign b_exp  = b[14:10];
    assign b_man  = (b_exp == 0) ? {1'b0, b[9:0]} : {1'b1, b[9:0]};

    wire [21:0] man_product;
    assign man_product = a_man * b_man;  // 11 x 11 = 22 bits

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 16'h0000;
            valid  <= 1'b0;
        end else if (enable) begin
            // Zero check
            if (a[14:0] == 15'd0 || b[14:0] == 15'd0) begin
                result <= 16'h0000;
            end
            // Inf/NaN check
            else if (a_exp == 5'h1F || b_exp == 5'h1F) begin
                result <= {a_sign ^ b_sign, 5'h1F, 10'd0}; // Inf
            end
            else begin
                // Sign: XOR
                // Exponent: add - bias
                // Mantissa: multiply and normalize
                reg        r_sign;
                reg [6:0]  r_exp_wide;
                reg [4:0]  r_exp;
                reg [9:0]  r_man;

                r_sign = a_sign ^ b_sign;
                r_exp_wide = {2'b0, a_exp} + {2'b0, b_exp} - 7'd15;

                if (man_product[21]) begin
                    // Product >= 2.0, shift right
                    r_man = man_product[20:11];
                    r_exp_wide = r_exp_wide + 1;
                end else begin
                    r_man = man_product[19:10];
                end

                // Clamp exponent
                if (r_exp_wide[6] || r_exp_wide == 0) begin
                    result <= 16'h0000; // Underflow
                end else if (r_exp_wide >= 7'd31) begin
                    result <= {r_sign, 5'h1F, 10'd0}; // Overflow -> Inf
                end else begin
                    r_exp = r_exp_wide[4:0];
                    result <= {r_sign, r_exp, r_man};
                end
            end
            valid <= 1'b1;
        end else begin
            valid <= 1'b0;
        end
    end

endmodule

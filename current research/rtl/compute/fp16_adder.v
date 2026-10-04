//============================================================================
// File:    fp16_adder.v
// Desc:    IEEE-754 Half-Precision (FP16) adder for Softmax computation
//          Format: 1-bit sign | 5-bit exponent | 10-bit mantissa
//============================================================================

module fp16_adder (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,
    input  wire [15:0] a,
    input  wire [15:0] b,
    output reg  [15:0] result,
    output reg         valid
);

    // Unpack inputs
    wire        a_sign, b_sign;
    wire [4:0]  a_exp,  b_exp;
    wire [10:0] a_man,  b_man;  // Includes implicit 1

    assign a_sign = a[15];
    assign a_exp  = a[14:10];
    assign a_man  = (a_exp == 0) ? {1'b0, a[9:0]} : {1'b1, a[9:0]};
    assign b_sign = b[15];
    assign b_exp  = b[14:10];
    assign b_man  = (b_exp == 0) ? {1'b0, b[9:0]} : {1'b1, b[9:0]};

    // Internal signals
    reg [4:0]  exp_diff;
    reg [4:0]  larger_exp;
    reg [11:0] man_a_aligned, man_b_aligned;
    reg [12:0] man_sum;
    reg        result_sign;
    reg [4:0]  result_exp;
    reg [9:0]  result_man;
    reg        pipe_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 16'h0000;
            valid  <= 1'b0;
        end else if (enable) begin
            // Handle zero inputs
            if (a[14:0] == 15'd0) begin
                result <= b;
                valid  <= 1'b1;
            end else if (b[14:0] == 15'd0) begin
                result <= a;
                valid  <= 1'b1;
            end else begin
                // Align mantissas to larger exponent
                if (a_exp >= b_exp) begin
                    larger_exp    = a_exp;
                    exp_diff      = a_exp - b_exp;
                    man_a_aligned = {1'b0, a_man};
                    man_b_aligned = (exp_diff < 12) ? ({1'b0, b_man} >> exp_diff) : 12'd0;
                end else begin
                    larger_exp    = b_exp;
                    exp_diff      = b_exp - a_exp;
                    man_a_aligned = (exp_diff < 12) ? ({1'b0, a_man} >> exp_diff) : 12'd0;
                    man_b_aligned = {1'b0, b_man};
                end

                // Add or subtract mantissas based on signs
                if (a_sign == b_sign) begin
                    man_sum     = {1'b0, man_a_aligned} + {1'b0, man_b_aligned};
                    result_sign = a_sign;
                end else begin
                    if (man_a_aligned >= man_b_aligned) begin
                        man_sum     = {1'b0, man_a_aligned} - {1'b0, man_b_aligned};
                        result_sign = a_sign;
                    end else begin
                        man_sum     = {1'b0, man_b_aligned} - {1'b0, man_a_aligned};
                        result_sign = b_sign;
                    end
                end

                // Normalize result
                if (man_sum == 0) begin
                    result <= 16'h0000;
                end else if (man_sum[12]) begin
                    // Overflow: shift right
                    result_exp = larger_exp + 1;
                    result_man = man_sum[11:2]; // Drop implicit 1 bit + round
                    result <= {result_sign, result_exp, result_man};
                end else if (man_sum[11]) begin
                    // Normal: already in place
                    result_exp = larger_exp;
                    result_man = man_sum[10:1];
                    result <= {result_sign, result_exp, result_man};
                end else begin
                    // Subnormal: find leading 1 and shift
                    if (man_sum[10])      begin result_exp = larger_exp - 1; result_man = man_sum[9:0]; end
                    else if (man_sum[9])  begin result_exp = larger_exp - 2; result_man = {man_sum[8:0], 1'b0}; end
                    else if (man_sum[8])  begin result_exp = larger_exp - 3; result_man = {man_sum[7:0], 2'b0}; end
                    else if (man_sum[7])  begin result_exp = larger_exp - 4; result_man = {man_sum[6:0], 3'b0}; end
                    else if (man_sum[6])  begin result_exp = larger_exp - 5; result_man = {man_sum[5:0], 4'b0}; end
                    else if (man_sum[5])  begin result_exp = larger_exp - 6; result_man = {man_sum[4:0], 5'b0}; end
                    else if (man_sum[4])  begin result_exp = larger_exp - 7; result_man = {man_sum[3:0], 6'b0}; end
                    else                  begin result_exp = 5'd0;           result_man = 10'd0; end
                    result <= {result_sign, result_exp, result_man};
                end

                valid <= 1'b1;
            end
        end else begin
            valid <= 1'b0;
        end
    end

endmodule

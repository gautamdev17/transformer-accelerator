//============================================================================
// File:    softmax_int.v
// Desc:    Stage 2 -- Integer-only Softmax (I-BERT style, no FP16 / no FPU)
//          Replaces softmax_fp16.v. Same handshake and BRAM interface; the
//          only interface change is attn_out: 16-bit FP16 -> 8-bit UINT8
//          (probability in Q0.8, i.e. real value = attn_out / 256).
//
//          Per row:
//            1. max   = max_j S[j]
//            2. x_j   = (S[j] - max) >>> 3            (<= 0, Q8.8 logit;
//                       this is the original "* 1/sqrt(dk)" scaling, dk=64)
//            3. i-exp (I-BERT, Kim et al. ICML'21), scale S_in = 2^-8:
//                 z   = floor(-x / ln2_q)             ln2_q = 177
//                 p   = x + z*ln2_q                   p in (-177, 0]
//                 L   = (p + 346)^2 + 62885           2nd-order poly of e^p
//                 exp = L >> z                        = e^x * 2^18 (approx)
//            4. sum  = sum_j exp_j
//            5. factor = floor(2^32 / sum)            (sequential divider)
//            6. A[j] = clamp((exp_j * factor + 2^23) >> 24, 0, 255)
//
//          Everything is add / shift / small integer multiply / one
//          sequential integer divide per row.  No floating-point unit.
//
//          Bit-exact reference: sim/softmax_int_ref.py
//============================================================================
`include "../common/defines.v"

module softmax_int #(
    parameter MAX_SEQ_LEN = `MAX_SEQ_LEN
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [6:0]  seq_len,

    // Input: INT32 raw scores (one row)
    output reg  [6:0]  score_rd_idx,
    input  wire signed [31:0] score_in,
    output reg         score_rd_en,

    // Output: UINT8 attention probabilities, Q0.8 (one row)
    output reg  [6:0]  attn_wr_idx,
    output reg  [7:0]  attn_out,
    output reg         attn_wr_en,

    // Row index for external address calculation
    output reg  [6:0]  current_row,

    output reg         done,
    output reg         busy
);

    // ---- I-BERT integer constants for input scale S = 2^-8 --------------
    //   Q_LN2 = floor(ln2 / S)             = 177
    //   Q_B   = floor(1.353 / S)           = 346
    //   Q_C   = floor(0.344 / (0.3585*S^2))= 62885
    // Max of L (p = 0) = 346^2 + 62885 = 182601  (< 2^18)
    localparam [8:0]  Q_LN2 = 9'd177;
    localparam [8:0]  Q_B   = 9'd346;
    localparam [16:0] Q_C   = 17'd62885;
    // u/177 == (u*5925)>>20 for all 0 <= u < 4096 (checked exhaustively)
    localparam [12:0] RECIP_LN2 = 13'd5925;

    // ---- FSM ----
    reg [3:0] state;
    localparam S_IDLE     = 4'd0;
    localparam S_FIND_MAX = 4'd1;
    localparam S_SCALE    = 4'd2;
    localparam S_EXP      = 4'd3;
    localparam S_SUM      = 4'd4;
    localparam S_DIV      = 4'd5;
    localparam S_NORMALIZE= 4'd6;
    localparam S_NEXT_ROW = 4'd7;
    localparam S_DONE     = 4'd8;

    // Row buffers
    reg signed [31:0] score_buf [0:MAX_SEQ_LEN-1];
    reg        [17:0] exp_buf   [0:MAX_SEQ_LEN-1];  // e^x * 2^18, UINT18
    reg [6:0]  idx;
    reg [6:0]  row_idx;

    // Intermediate values
    reg signed [31:0] max_score;
    reg        [11:0] neg_x;      // -x, saturated to 12 bits (z>=18 => exp=0)
    reg        [25:0] exp_sum;    // <= 128 * 2^18 < 2^26
    reg        [15:0] factor;     // floor(2^32/sum) <= 23521 (sum >= 182601)

    // ---- combinational i-exp ---------------------------------------------
    wire signed [32:0] diff_w  = $signed({score_buf[idx][31], score_buf[idx]})
                               - $signed({max_score[31], max_score});   // <= 0
    wire        [29:0] diff_abs = -diff_w[32:3];                        // = -(diff>>>3)

    wire [24:0] z_mul   = neg_x * RECIP_LN2;
    wire [4:0]  z       = z_mul[24:20];                // floor(neg_x / 177)
    wire [16:0] zq      = z * Q_LN2;                   // z * 177
    // p = x + z*ln2 = zq - neg_x  (<= 0)  =>  p + Q_B = Q_B - (neg_x - zq)
    wire [8:0]  r       = neg_x - zq;                  // -p, in [0, 176]
    wire [8:0]  t_pb    = Q_B - r;                     // p + Q_B, in [170, 346]
    wire [17:0] poly    = t_pb * t_pb + Q_C;           // L
    wire [17:0] exp_val = (z >= 5'd18) ? 18'd0 : (poly >> z);

    // ---- sequential restoring divider: factor = floor(2^32 / exp_sum) ----
    reg [5:0]  div_cnt;                // 32 .. 0  (33 iterations)
    reg [26:0] div_rem;
    wire [26:0] div_shift = {div_rem[25:0], (div_cnt == 6'd32)};  // numerator = 2^32
    wire        div_ge    = (div_shift >= {1'b0, exp_sum});

    // ---- normalise: A = clamp((exp * factor + 2^23) >> 24) ---------------
    wire [33:0] norm_prod = exp_buf[idx] * factor;
    wire [33:0] norm_rnd  = norm_prod + 34'd8388608;
    wire [9:0]  norm_q    = norm_rnd[33:24];
    wire [7:0]  norm_sat  = (norm_q > 10'd255) ? 8'd255 : norm_q[7:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            done        <= 1'b0;
            busy        <= 1'b0;
            score_rd_en <= 1'b0;
            attn_wr_en  <= 1'b0;
            row_idx     <= 0;
            current_row <= 0;
        end else begin
            // Default deasserts
            score_rd_en <= 1'b0;
            attn_wr_en  <= 1'b0;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy         <= 1'b1;
                        row_idx      <= 0;
                        current_row  <= 0;
                        idx          <= 0;
                        state        <= S_FIND_MAX;
                        score_rd_en  <= 1'b1;
                        score_rd_idx <= 0;
                        max_score    <= 32'sh80000000; // INT32_MIN
                    end
                end

                S_FIND_MAX: begin
                    // Read scores and find max for numerical stability
                    score_rd_en  <= 1'b1;
                    score_rd_idx <= idx + 1;

                    score_buf[idx] <= score_in;
                    if ($signed(score_in) > $signed(max_score))
                        max_score <= score_in;

                    if (idx == seq_len - 1) begin
                        idx         <= 0;
                        state       <= S_SCALE;
                        score_rd_en <= 1'b0;
                    end else begin
                        idx <= idx + 1;
                    end
                end

                S_SCALE: begin
                    // x = (score - max) / 8 (Q8.8, <= 0); keep -x, saturated
                    neg_x <= (diff_abs > 30'd4095) ? 12'd4095 : diff_abs[11:0];
                    state <= S_EXP;
                end

                S_EXP: begin
                    // integer exp via I-BERT 2nd-order polynomial + shift
                    exp_buf[idx] <= exp_val;

                    if (idx == seq_len - 1) begin
                        idx     <= 0;
                        exp_sum <= 26'd0;
                        state   <= S_SUM;
                    end else begin
                        idx   <= idx + 1;
                        state <= S_SCALE;
                    end
                end

                S_SUM: begin
                    exp_sum <= exp_sum + exp_buf[idx];
                    if (idx == seq_len - 1) begin
                        idx     <= 0;
                        div_cnt <= 6'd32;
                        div_rem <= 27'd0;
                        factor  <= 16'd0;
                        state   <= S_DIV;
                    end else begin
                        idx <= idx + 1;
                    end
                end

                S_DIV: begin
                    // restoring division, one numerator bit per cycle
                    if (div_ge) begin
                        div_rem <= div_shift - {1'b0, exp_sum};
                        factor  <= {factor[14:0], 1'b1};
                    end else begin
                        div_rem <= div_shift;
                        factor  <= {factor[14:0], 1'b0};
                    end
                    if (div_cnt == 6'd0) begin
                        idx   <= 0;
                        state <= S_NORMALIZE;
                    end else begin
                        div_cnt <= div_cnt - 1;
                    end
                end

                S_NORMALIZE: begin
                    // attn[i] = exp[i] * factor >> 24 (rounded, saturated to 255)
                    attn_out    <= norm_sat;
                    attn_wr_idx <= idx;
                    attn_wr_en  <= 1'b1;

                    if (idx == seq_len - 1) begin
                        state <= S_NEXT_ROW;
                    end else begin
                        idx <= idx + 1;
                    end
                end

                S_NEXT_ROW: begin
                    if (row_idx == seq_len - 1) begin
                        state <= S_DONE;
                    end else begin
                        row_idx      <= row_idx + 1;
                        current_row  <= row_idx + 1;
                        idx          <= 0;
                        max_score    <= 32'sh80000000;
                        score_rd_en  <= 1'b1;
                        score_rd_idx <= 0;
                        state        <= S_FIND_MAX;
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

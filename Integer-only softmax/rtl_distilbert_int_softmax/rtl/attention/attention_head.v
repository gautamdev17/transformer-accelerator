//============================================================================
// File:    attention_head.v
// Desc:    Single Attention Head — orchestrates the 3-stage pipeline:
//          Stage 1: QK^T (INT8) → Stage 2: Softmax (integer-only) → Stage 3: AV (INT4)
//============================================================================
`include "../common/defines.v"

module attention_head #(
    parameter D_K       = `D_K,
    parameter TILE_SIZE = `TILE_SIZE,
    parameter NUM_PES   = `NUM_PES
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [6:0]  seq_len,

    // Q matrix BRAM interface
    output wire [17:0] q_addr,
    input  wire [63:0] q_data,
    output wire        q_rd_en,

    // K^T matrix BRAM interface
    output wire [17:0] kt_addr,
    input  wire [63:0] kt_data,
    output wire        kt_rd_en,

    // V matrix BRAM interface
    output wire [17:0] v_addr,
    input  wire [63:0] v_data,
    output wire        v_rd_en,

    // Output context BRAM interface
    output wire [17:0] ctx_addr,
    output wire [31:0] ctx_data,
    output wire        ctx_wr_en,

    output wire        done,
    output wire        busy
);

    // ---- FSM ----
    reg [2:0] state;
    localparam S_IDLE    = 3'd0;
    localparam S_QK      = 3'd1;
    localparam S_SOFTMAX = 3'd2;
    localparam S_AV      = 3'd3;
    localparam S_DONE    = 3'd4;

    // Stage control signals
    reg  qk_start, sm_start, av_start;
    wire qk_done, qk_busy;
    wire sm_done, sm_busy;
    wire av_done, av_busy;

    // Internal score BRAM (S = QK^T scores, seq_len × seq_len, INT32)
    reg  [17:0] score_wr_addr, score_rd_addr;
    reg  [31:0] score_wr_data;
    reg         score_wr_en;
    wire [31:0] score_rd_data;

    // Use QK output to write scores
    wire [17:0] qk_score_addr;
    wire [31:0] qk_score_data;
    wire        qk_score_wr;

    // Attention weight buffer (A = softmax output, UINT8 Q0.8)
    wire [17:0] sm_attn_addr;
    wire [7:0]  sm_attn_data;
    wire        sm_attn_wr;
    wire [6:0]  sm_row;
    wire [6:0]  sm_score_idx;
    wire        sm_score_rd;

    // AV attention read
    wire [17:0] av_attn_addr;
    wire [7:0]  av_attn_data;
    wire        av_attn_rd;

    // Score BRAM (dual-port: write from QK, read from Softmax)
    reg [31:0] score_mem [0:16383]; // seq_len^2 max = 128*128 = 16384

    always @(posedge clk) begin
        if (qk_score_wr)
            score_mem[qk_score_addr[13:0]] <= qk_score_data;
    end

    assign score_rd_data = score_mem[{sm_row, sm_score_idx}];

    // Attention weight BRAM (dual-port: write from Softmax, read from AV)
    reg [7:0] attn_mem [0:16383];
    wire [6:0] sm_wr_idx;

    always @(posedge clk) begin
        if (sm_attn_wr)
            attn_mem[{sm_row, sm_wr_idx}] <= sm_attn_data;
    end

    assign av_attn_data = attn_mem[av_attn_addr[13:0]];

    // ---- Stage 1: QK Matmul ----
    qk_matmul_int8 #(
        .TILE_SIZE(TILE_SIZE), .D_K(D_K), .NUM_PES(NUM_PES)
    ) u_qk (
        .clk(clk), .rst_n(rst_n),
        .start(qk_start), .seq_len(seq_len),
        .q_addr(q_addr), .q_data(q_data), .q_rd_en(q_rd_en),
        .kt_addr(kt_addr), .kt_data(kt_data), .kt_rd_en(kt_rd_en),
        .score_addr(qk_score_addr), .score_data(qk_score_data),
        .score_wr_en(qk_score_wr),
        .done(qk_done), .busy(qk_busy)
    );

    // ---- Stage 2: Softmax ----
    softmax_int u_softmax (
        .clk(clk), .rst_n(rst_n),
        .start(sm_start), .seq_len(seq_len),
        .score_rd_idx(sm_score_idx), .score_in(score_rd_data),
        .score_rd_en(sm_score_rd),
        .attn_wr_idx(sm_wr_idx), .attn_out(sm_attn_data),
        .attn_wr_en(sm_attn_wr),
        .current_row(sm_row),
        .done(sm_done), .busy(sm_busy)
    );

    // ---- Stage 3: AV Matmul ----
    av_matmul_int4 #(
        .TILE_SIZE(TILE_SIZE), .D_K(D_K), .NUM_PES(NUM_PES)
    ) u_av (
        .clk(clk), .rst_n(rst_n),
        .start(av_start), .seq_len(seq_len),
        .attn_addr(av_attn_addr), .attn_data(av_attn_data),
        .attn_rd_en(av_attn_rd),
        .v_addr(v_addr), .v_data(v_data), .v_rd_en(v_rd_en),
        .ctx_addr(ctx_addr), .ctx_data(ctx_data), .ctx_wr_en(ctx_wr_en),
        .done(av_done), .busy(av_busy)
    );

    // ---- Head FSM ----
    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            qk_start <= 1'b0;
            sm_start <= 1'b0;
            av_start <= 1'b0;
        end else begin
            qk_start <= 1'b0;
            sm_start <= 1'b0;
            av_start <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        state    <= S_QK;
                        qk_start <= 1'b1;
                    end
                end

                S_QK: begin
                    if (qk_done) begin
                        state    <= S_SOFTMAX;
                        sm_start <= 1'b1;
                    end
                end

                S_SOFTMAX: begin
                    if (sm_done) begin
                        state    <= S_AV;
                        av_start <= 1'b1;
                    end
                end

                S_AV: begin
                    if (av_done) begin
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

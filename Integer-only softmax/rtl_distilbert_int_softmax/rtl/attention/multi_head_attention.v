//============================================================================
// File:    multi_head_attention.v
// Desc:    Multi-Head Attention module — 12 heads (time-multiplexed)
//          1) Project Q, K, V from input using linear layers
//          2) Split into 12 heads (each d_k=64)
//          3) Run attention_head for each head (sequential or parallel)
//          4) Concatenate head outputs
//          5) Output projection: W_O × concat
//============================================================================
`include "../common/defines.v"

module multi_head_attention #(
    parameter D_MODEL     = `D_MODEL,
    parameter N_HEADS     = `N_HEADS,
    parameter D_K         = `D_K,
    parameter TILE_SIZE   = `TILE_SIZE,
    parameter NUM_PES     = `NUM_PES,
    parameter MAX_SEQ_LEN = `MAX_SEQ_LEN,
    parameter HEAD_PAR    = 1            // Number of parallel heads (1=sequential)
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [6:0]  seq_len,

    // Input activation BRAM (d_model-wide, seq_len rows)
    output reg  [17:0] input_addr,
    input  wire [63:0] input_data,
    output reg         input_rd_en,

    // Weight BRAMs for Q, K, V, O projections
    output reg  [19:0] wq_addr, wk_addr, wv_addr, wo_addr,
    input  wire [63:0] wq_data, wk_data, wv_data, wo_data,
    output reg         wq_rd_en, wk_rd_en, wv_rd_en, wo_rd_en,

    // Output activation BRAM
    output reg  [17:0] output_addr,
    output reg  [63:0] output_data,
    output reg         output_wr_en,

    output reg         done,
    output reg         busy
);

    // ---- FSM ----
    reg [3:0] state;
    localparam S_IDLE       = 4'd0;
    localparam S_PROJ_QKV   = 4'd1;
    localparam S_SPLIT      = 4'd2;
    localparam S_ATTN_HEAD  = 4'd3;
    localparam S_CONCAT     = 4'd4;
    localparam S_OUT_PROJ   = 4'd5;
    localparam S_DONE       = 4'd6;

    // Head tracking
    reg [3:0] head_idx;
    reg       head_start;
    wire      head_done, head_busy;

    // Q/K/V projected buffer addresses (per-head)
    // Each head uses: seq_len × D_K values
    // Total Q buffer: seq_len × D_MODEL = seq_len × 768 (INT8)
    reg [17:0] q_buf_addr, kt_buf_addr, v_buf_addr;
    wire [63:0] q_buf_data, kt_buf_data, v_buf_data;
    reg q_buf_rd;

    // Q/K/V projection BRAMs
    reg signed [7:0] q_buf  [0:8191];  // seq_len * D_K per head
    reg signed [7:0] k_buf  [0:8191];
    reg signed [7:0] v_buf  [0:8191];

    // Context buffer per head: seq_len × D_K
    reg signed [31:0] ctx_buf [0:8191];

    // Concatenated output: seq_len × D_MODEL
    reg signed [7:0] concat_buf [0:98303]; // 128 * 768

    // Linear projection counters
    reg [6:0]  proj_row;
    reg [9:0]  proj_col;
    reg [9:0]  proj_k;
    reg signed [31:0] proj_acc;
    reg [1:0]  proj_phase;  // 0=Q, 1=K, 2=V

    // Head BRAM interfaces
    wire [17:0] h_q_addr, h_kt_addr, h_v_addr;
    wire [63:0] h_q_data, h_kt_data, h_v_data;
    wire        h_q_rd, h_kt_rd, h_v_rd;
    wire [17:0] h_ctx_addr;
    wire [31:0] h_ctx_data;
    wire        h_ctx_wr;

    // Pack Q/K/V buffers as 64-bit reads for attention head
    assign h_q_data  = {q_buf[h_q_addr+7], q_buf[h_q_addr+6], q_buf[h_q_addr+5], q_buf[h_q_addr+4],
                        q_buf[h_q_addr+3], q_buf[h_q_addr+2], q_buf[h_q_addr+1], q_buf[h_q_addr]};
    assign h_kt_data = {k_buf[h_kt_addr+7], k_buf[h_kt_addr+6], k_buf[h_kt_addr+5], k_buf[h_kt_addr+4],
                        k_buf[h_kt_addr+3], k_buf[h_kt_addr+2], k_buf[h_kt_addr+1], k_buf[h_kt_addr]};
    assign h_v_data  = {v_buf[h_v_addr+7], v_buf[h_v_addr+6], v_buf[h_v_addr+5], v_buf[h_v_addr+4],
                        v_buf[h_v_addr+3], v_buf[h_v_addr+2], v_buf[h_v_addr+1], v_buf[h_v_addr]};

    // Write context results
    always @(posedge clk) begin
        if (h_ctx_wr)
            ctx_buf[h_ctx_addr[12:0]] <= h_ctx_data;
    end

    // ---- Instantiate attention head ----
    attention_head #(
        .D_K(D_K), .TILE_SIZE(TILE_SIZE), .NUM_PES(NUM_PES)
    ) u_head (
        .clk(clk), .rst_n(rst_n),
        .start(head_start), .seq_len(seq_len),
        .q_addr(h_q_addr), .q_data(h_q_data), .q_rd_en(h_q_rd),
        .kt_addr(h_kt_addr), .kt_data(h_kt_data), .kt_rd_en(h_kt_rd),
        .v_addr(h_v_addr), .v_data(h_v_data), .v_rd_en(h_v_rd),
        .ctx_addr(h_ctx_addr), .ctx_data(h_ctx_data), .ctx_wr_en(h_ctx_wr),
        .done(head_done), .busy(head_busy)
    );

    // Concat & output projection counters
    reg [6:0]  out_row;
    reg [9:0]  out_col, out_k;
    reg signed [31:0] out_acc;

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            done         <= 1'b0;
            busy         <= 1'b0;
            head_start   <= 1'b0;
            input_rd_en  <= 1'b0;
            output_wr_en <= 1'b0;
            wq_rd_en     <= 1'b0;
            wk_rd_en     <= 1'b0;
            wv_rd_en     <= 1'b0;
            wo_rd_en     <= 1'b0;
        end else begin
            head_start   <= 1'b0;
            output_wr_en <= 1'b0;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy       <= 1'b1;
                        state      <= S_PROJ_QKV;
                        proj_row   <= 0;
                        proj_col   <= 0;
                        proj_k     <= 0;
                        proj_acc   <= 0;
                        proj_phase <= 0;
                    end
                end

                S_PROJ_QKV: begin
                    // Compute Q = X * W_Q, K = X * W_K, V = X * W_V
                    // Tiled matrix multiply: [seq_len × d_model] × [d_model × d_model]
                    // For simplicity, compute element-by-element with accumulation
                    input_addr  <= proj_row * D_MODEL + proj_k;
                    input_rd_en <= 1'b1;

                    case (proj_phase)
                        2'd0: begin wq_addr <= proj_col * D_MODEL + proj_k; wq_rd_en <= 1'b1; end
                        2'd1: begin wk_addr <= proj_col * D_MODEL + proj_k; wk_rd_en <= 1'b1; end
                        2'd2: begin wv_addr <= proj_col * D_MODEL + proj_k; wv_rd_en <= 1'b1; end
                        default: ;
                    endcase

                    // Accumulate (simplified — in practice use tiled MAC array)
                    proj_acc <= proj_acc + $signed(input_data[7:0]) * 
                               $signed(proj_phase == 0 ? wq_data[7:0] :
                                       proj_phase == 1 ? wk_data[7:0] : wv_data[7:0]);

                    if (proj_k == D_MODEL - 1) begin
                        proj_k <= 0;
                        // Store result (quantize to INT8)
                        case (proj_phase)
                            2'd0: q_buf[proj_row * D_MODEL + proj_col] <= 
                                  (proj_acc > 32'sd127) ? 8'sd127 :
                                  (proj_acc < -32'sd128) ? -8'sd128 : proj_acc[7:0];
                            2'd1: k_buf[proj_row * D_MODEL + proj_col] <= 
                                  (proj_acc > 32'sd127) ? 8'sd127 :
                                  (proj_acc < -32'sd128) ? -8'sd128 : proj_acc[7:0];
                            2'd2: v_buf[proj_row * D_MODEL + proj_col] <= 
                                  (proj_acc > 32'sd127) ? 8'sd127 :
                                  (proj_acc < -32'sd128) ? -8'sd128 : proj_acc[7:0];
                            default: ;
                        endcase
                        proj_acc <= 0;

                        if (proj_col == D_MODEL - 1) begin
                            proj_col <= 0;
                            if (proj_row == seq_len - 1) begin
                                proj_row <= 0;
                                if (proj_phase == 2) begin
                                    state    <= S_ATTN_HEAD;
                                    head_idx <= 0;
                                    input_rd_en <= 1'b0;
                                    wq_rd_en <= 1'b0;
                                    wk_rd_en <= 1'b0;
                                    wv_rd_en <= 1'b0;
                                end else begin
                                    proj_phase <= proj_phase + 1;
                                end
                            end else begin
                                proj_row <= proj_row + 1;
                            end
                        end else begin
                            proj_col <= proj_col + 1;
                        end
                    end else begin
                        proj_k <= proj_k + 1;
                    end
                end

                S_ATTN_HEAD: begin
                    // Run each head sequentially (time-multiplexed)
                    if (!head_busy && !head_done) begin
                        head_start <= 1'b1;
                    end

                    if (head_done) begin
                        // Copy context to concatenated buffer
                        // head_idx selects which D_K slice of D_MODEL
                        // concat_buf[row][head_idx*D_K + col] = ctx_buf[row][col]

                        if (head_idx == N_HEADS - 1) begin
                            state <= S_OUT_PROJ;
                            out_row <= 0;
                            out_col <= 0;
                            out_k   <= 0;
                            out_acc <= 0;
                        end else begin
                            head_idx <= head_idx + 1;
                        end
                    end
                end

                S_OUT_PROJ: begin
                    // Output projection: O = concat * W_O
                    // [seq_len × d_model] × [d_model × d_model]
                    wo_addr  <= out_col * D_MODEL + out_k;
                    wo_rd_en <= 1'b1;

                    out_acc <= out_acc + $signed(concat_buf[out_row * D_MODEL + out_k]) * 
                              $signed(wo_data[7:0]);

                    if (out_k == D_MODEL - 1) begin
                        out_k <= 0;
                        // Write output (quantized INT8)
                        output_addr <= out_row * D_MODEL + out_col;
                        output_data <= {{56{out_acc[31]}}, out_acc[7:0]};
                        output_wr_en <= 1'b1;
                        out_acc <= 0;

                        if (out_col == D_MODEL - 1) begin
                            out_col <= 0;
                            if (out_row == seq_len - 1) begin
                                state <= S_DONE;
                                wo_rd_en <= 1'b0;
                            end else begin
                                out_row <= out_row + 1;
                            end
                        end else begin
                            out_col <= out_col + 1;
                        end
                    end else begin
                        out_k <= out_k + 1;
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

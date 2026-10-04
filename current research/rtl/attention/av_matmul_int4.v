//============================================================================
// File:    av_matmul_int4.v
// Desc:    Stage 3 — Attention-Value Projection Engine (INT4 precision)
//          Computes C[i][k] = sum_j A[i][j] * V[j][k]
//          A = attention weights (FP16 from softmax, cast to INT4)
//          V = value matrix (pre-quantized to INT4)
//          Tile-based with INT4 SIMD MAC units
//============================================================================
`include "../common/defines.v"

module av_matmul_int4 #(
    parameter TILE_SIZE = `TILE_SIZE,
    parameter D_K       = `D_K,
    parameter NUM_PES   = `NUM_PES
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [6:0]  seq_len,

    // Attention weights (FP16 from softmax — will be cast to INT4)
    output reg  [17:0] attn_addr,
    input  wire [15:0] attn_data,        // FP16 single value
    output reg         attn_rd_en,

    // Value matrix (INT4, packed: 16 INT4 values per 64-bit word)
    output reg  [17:0] v_addr,
    input  wire [63:0] v_data,
    output reg         v_rd_en,

    // Output context vector (INT32 accumulation)
    output reg  [17:0] ctx_addr,
    output reg  [31:0] ctx_data,
    output reg         ctx_wr_en,

    output reg         done,
    output reg         busy
);

    // ---- FSM ----
    reg [3:0] state;
    localparam S_IDLE      = 4'd0;
    localparam S_LOAD_A    = 4'd1;
    localparam S_LOAD_V    = 4'd2;
    localparam S_COMPUTE   = 4'd3;
    localparam S_STORE     = 4'd4;
    localparam S_NEXT_TILE = 4'd5;
    localparam S_DONE      = 4'd6;

    // Tile tracking
    reg [6:0] tile_row, tile_col;
    reg [3:0] row_in_tile, col_in_tile;
    reg [6:0] k_idx;
    reg [6:0] k_tile_idx;

    // Tile buffers
    reg signed [3:0]  a_tile [0:TILE_SIZE-1][0:TILE_SIZE-1];
    reg signed [3:0]  v_tile [0:TILE_SIZE-1][0:TILE_SIZE-1];
    reg signed [31:0] acc    [0:TILE_SIZE-1][0:TILE_SIZE-1];

    // MAC array
    wire signed [31:0] mac_acc [0:NUM_PES-1];
    reg         mac_clear [0:NUM_PES-1];
    reg         mac_enable [0:NUM_PES-1];
    reg  signed [3:0] mac_a [0:NUM_PES-1];
    reg  signed [3:0] mac_b [0:NUM_PES-1];

    genvar g;
    generate
        for (g = 0; g < NUM_PES; g = g + 1) begin : mac_array
            mac_unit_int4 u_mac (
                .clk(clk), .rst_n(rst_n),
                .clear(mac_clear[g]),
                .enable(mac_enable[g]),
                .a(mac_a[g]),
                .b(mac_b[g]),
                .acc(mac_acc[g])
            );
        end
    endgenerate

    // FP16 to INT4 conversion for attention weights
    wire signed [3:0] attn_int4;
    fp16_to_int4 u_fp16_to_int4 (
        .in_fp16(attn_data),
        .scale(8'd16),           // Scale factor for INT4 range
        .out_int4(attn_int4)
    );

    // Load counters
    reg [3:0] load_row, load_col;

    integer i, j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            done       <= 1'b0;
            busy       <= 1'b0;
            attn_rd_en <= 1'b0;
            v_rd_en    <= 1'b0;
            ctx_wr_en  <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    ctx_wr_en <= 1'b0;
                    if (start) begin
                        busy       <= 1'b1;
                        tile_row   <= 0;
                        tile_col   <= 0;
                        k_tile_idx <= 0;
                        for (i = 0; i < TILE_SIZE; i = i + 1)
                            for (j = 0; j < TILE_SIZE; j = j + 1)
                                acc[i][j] <= 32'sd0;
                        state    <= S_LOAD_A;
                        load_row <= 0;
                        load_col <= 0;
                    end
                end

                S_LOAD_A: begin
                    // Read attention weights (FP16) and convert to INT4
                    attn_addr  <= (tile_row * TILE_SIZE + load_row) * seq_len + 
                                  k_tile_idx * TILE_SIZE + load_col;
                    attn_rd_en <= 1'b1;

                    if (attn_rd_en) begin
                        a_tile[load_row][load_col] <= attn_int4;

                        if (load_col == TILE_SIZE - 1) begin
                            load_col <= 0;
                            if (load_row == TILE_SIZE - 1) begin
                                load_row   <= 0;
                                attn_rd_en <= 1'b0;
                                state      <= S_LOAD_V;
                            end else begin
                                load_row <= load_row + 1;
                            end
                        end else begin
                            load_col <= load_col + 1;
                        end
                    end
                end

                S_LOAD_V: begin
                    // Read V matrix values (INT4, packed in 64-bit words)
                    v_addr  <= (k_tile_idx * TILE_SIZE + load_row) * D_K + tile_col * TILE_SIZE + load_col;
                    v_rd_en <= 1'b1;

                    if (v_rd_en) begin
                        // Unpack INT4 values (16 per 64-bit word, use first TILE_SIZE)
                        for (i = 0; i < 8 && (load_col + i) < TILE_SIZE; i = i + 1)
                            v_tile[load_row][load_col + i] <= v_data[i*4 +: 4];

                        if (load_col + 8 >= TILE_SIZE) begin
                            load_col <= 0;
                            if (load_row == TILE_SIZE - 1) begin
                                load_row <= 0;
                                v_rd_en  <= 1'b0;
                                state    <= S_COMPUTE;
                                row_in_tile <= 0;
                                col_in_tile <= 0;
                                k_idx <= 0;
                            end else begin
                                load_row <= load_row + 1;
                            end
                        end else begin
                            load_col <= load_col + 8;
                        end
                    end
                end

                S_COMPUTE: begin
                    for (i = 0; i < NUM_PES && (col_in_tile + i) < TILE_SIZE; i = i + 1) begin
                        mac_a[i]      <= a_tile[row_in_tile][k_idx];
                        mac_b[i]      <= v_tile[k_idx][col_in_tile + i];
                        mac_enable[i] <= 1'b1;
                        mac_clear[i]  <= (k_idx == 0 && k_tile_idx == 0) ? 1'b1 : 1'b0;
                    end

                    if (k_idx == TILE_SIZE - 1) begin
                        k_idx <= 0;
                        for (i = 0; i < NUM_PES && (col_in_tile + i) < TILE_SIZE; i = i + 1) begin
                            acc[row_in_tile][col_in_tile + i] <= acc[row_in_tile][col_in_tile + i] + mac_acc[i];
                            mac_enable[i] <= 1'b0;
                        end

                        if (col_in_tile + NUM_PES >= TILE_SIZE) begin
                            col_in_tile <= 0;
                            if (row_in_tile == TILE_SIZE - 1) begin
                                row_in_tile <= 0;
                                if ((k_tile_idx + 1) * TILE_SIZE < seq_len) begin
                                    k_tile_idx <= k_tile_idx + 1;
                                    state <= S_LOAD_A;
                                    load_row <= 0;
                                    load_col <= 0;
                                end else begin
                                    state <= S_STORE;
                                    load_row <= 0;
                                    load_col <= 0;
                                end
                            end else begin
                                row_in_tile <= row_in_tile + 1;
                            end
                        end else begin
                            col_in_tile <= col_in_tile + NUM_PES;
                        end
                    end else begin
                        k_idx <= k_idx + 1;
                    end
                end

                S_STORE: begin
                    ctx_addr  <= (tile_row * TILE_SIZE + load_row) * D_K + 
                                 (tile_col * TILE_SIZE + load_col);
                    ctx_data  <= acc[load_row][load_col];
                    ctx_wr_en <= 1'b1;

                    if (load_col == TILE_SIZE - 1) begin
                        load_col <= 0;
                        if (load_row == TILE_SIZE - 1) begin
                            ctx_wr_en <= 1'b0;
                            state     <= S_NEXT_TILE;
                        end else begin
                            load_row <= load_row + 1;
                        end
                    end else begin
                        load_col <= load_col + 1;
                    end
                end

                S_NEXT_TILE: begin
                    k_tile_idx <= 0;
                    for (i = 0; i < TILE_SIZE; i = i + 1)
                        for (j = 0; j < TILE_SIZE; j = j + 1)
                            acc[i][j] <= 32'sd0;

                    if (tile_col + 1 < (D_K + TILE_SIZE - 1) / TILE_SIZE) begin
                        tile_col <= tile_col + 1;
                        state    <= S_LOAD_A;
                        load_row <= 0;
                        load_col <= 0;
                    end else begin
                        tile_col <= 0;
                        if (tile_row + 1 < (seq_len + TILE_SIZE - 1) / TILE_SIZE) begin
                            tile_row <= tile_row + 1;
                            state    <= S_LOAD_A;
                            load_row <= 0;
                            load_col <= 0;
                        end else begin
                            state <= S_DONE;
                        end
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

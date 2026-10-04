//============================================================================
// File:    qk_matmul_int8.v
// Desc:    Stage 1 — QK^T Dot-Product Engine (INT8 precision)
//          Computes S[i][j] = sum_k Q[i][k] * K[j][k] for one tile
//          Uses NUM_PES parallel MAC units for throughput
//          Tile-based: processes TILE_SIZE x TILE_SIZE output tiles
//============================================================================
`include "../common/defines.v"

module qk_matmul_int8 #(
    parameter TILE_SIZE = `TILE_SIZE,
    parameter D_K       = `D_K,
    parameter NUM_PES   = `NUM_PES
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [6:0]  seq_len,          // Actual sequence length

    // Q matrix interface (row-major, INT8)
    output reg  [17:0] q_addr,
    input  wire [63:0] q_data,           // 8 x INT8 packed
    output reg         q_rd_en,

    // K^T matrix interface (col-major, INT8)
    output reg  [17:0] kt_addr,
    input  wire [63:0] kt_data,
    output reg         kt_rd_en,

    // Output score interface
    output reg  [17:0] score_addr,
    output reg  [31:0] score_data,       // INT32 raw score
    output reg         score_wr_en,

    output reg         done,
    output reg         busy
);

    // ---- Internal state ----
    reg [3:0]  state;
    localparam S_IDLE      = 4'd0;
    localparam S_LOAD_Q    = 4'd1;
    localparam S_LOAD_KT   = 4'd2;
    localparam S_COMPUTE   = 4'd3;
    localparam S_STORE     = 4'd4;
    localparam S_NEXT_TILE = 4'd5;
    localparam S_DONE      = 4'd6;

    // Tile position tracking
    reg [6:0] tile_row, tile_col;        // Current tile position (in tile units)
    reg [3:0] row_in_tile, col_in_tile;  // Position within tile
    reg [6:0] k_idx;                     // Inner dimension counter

    // Local buffers for current tile
    reg signed [7:0] q_tile  [0:TILE_SIZE-1][0:TILE_SIZE-1];  // Q tile
    reg signed [7:0] kt_tile [0:TILE_SIZE-1][0:TILE_SIZE-1];  // K^T tile
    reg signed [31:0] acc    [0:TILE_SIZE-1][0:TILE_SIZE-1];   // Accumulator

    // MAC array — instantiate NUM_PES parallel MAC units
    wire signed [31:0] mac_acc [0:NUM_PES-1];
    reg         mac_clear [0:NUM_PES-1];
    reg         mac_enable [0:NUM_PES-1];
    reg  signed [7:0] mac_a [0:NUM_PES-1];
    reg  signed [7:0] mac_b [0:NUM_PES-1];

    genvar g;
    generate
        for (g = 0; g < NUM_PES; g = g + 1) begin : mac_array
            mac_unit_int8 u_mac (
                .clk(clk), .rst_n(rst_n),
                .clear(mac_clear[g]),
                .enable(mac_enable[g]),
                .a(mac_a[g]),
                .b(mac_b[g]),
                .acc(mac_acc[g])
            );
        end
    endgenerate

    // Load counters
    reg [3:0] load_row, load_col;
    reg [6:0] k_tile_idx;                // Which d_k tile we're on
    reg [3:0] byte_idx;

    integer i, j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            done        <= 1'b0;
            busy        <= 1'b0;
            q_rd_en     <= 1'b0;
            kt_rd_en    <= 1'b0;
            score_wr_en <= 1'b0;
            tile_row    <= 0;
            tile_col    <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    score_wr_en <= 1'b0;
                    if (start) begin
                        busy     <= 1'b1;
                        tile_row <= 0;
                        tile_col <= 0;
                        // Clear all accumulators
                        for (i = 0; i < TILE_SIZE; i = i + 1)
                            for (j = 0; j < TILE_SIZE; j = j + 1)
                                acc[i][j] <= 32'sd0;
                        k_tile_idx <= 0;
                        state    <= S_LOAD_Q;
                        load_row <= 0;
                        load_col <= 0;
                    end
                end

                S_LOAD_Q: begin
                    // Load one row of Q tile from BRAM (8 bytes per read)
                    q_addr  <= (tile_row * TILE_SIZE + load_row) * D_K + k_tile_idx * TILE_SIZE + load_col;
                    q_rd_en <= 1'b1;

                    // Unpack 8 INT8 values from 64-bit bus
                    if (q_rd_en) begin
                        for (i = 0; i < 8 && (load_col + i) < TILE_SIZE; i = i + 1)
                            q_tile[load_row][load_col + i] <= q_data[i*8 +: 8];

                        if (load_col + 8 >= TILE_SIZE) begin
                            load_col <= 0;
                            if (load_row == TILE_SIZE - 1) begin
                                load_row <= 0;
                                q_rd_en  <= 1'b0;
                                state    <= S_LOAD_KT;
                            end else begin
                                load_row <= load_row + 1;
                            end
                        end else begin
                            load_col <= load_col + 8;
                        end
                    end
                end

                S_LOAD_KT: begin
                    // Load K^T tile (transposed K)
                    kt_addr  <= (tile_col * TILE_SIZE + load_row) * D_K + k_tile_idx * TILE_SIZE + load_col;
                    kt_rd_en <= 1'b1;

                    if (kt_rd_en) begin
                        for (i = 0; i < 8 && (load_col + i) < TILE_SIZE; i = i + 1)
                            kt_tile[load_row][load_col + i] <= kt_data[i*8 +: 8];

                        if (load_col + 8 >= TILE_SIZE) begin
                            load_col <= 0;
                            if (load_row == TILE_SIZE - 1) begin
                                load_row <= 0;
                                kt_rd_en <= 1'b0;
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
                    // Compute: for each (i,j) in tile, accumulate over k dimension
                    // Use NUM_PES MACs to compute NUM_PES output elements in parallel
                    for (i = 0; i < NUM_PES && (col_in_tile + i) < TILE_SIZE; i = i + 1) begin
                        mac_a[i]      <= q_tile[row_in_tile][k_idx];
                        mac_b[i]      <= kt_tile[col_in_tile + i][k_idx];
                        mac_enable[i] <= 1'b1;
                        mac_clear[i]  <= (k_idx == 0 && k_tile_idx == 0) ? 1'b1 : 1'b0;
                    end

                    if (k_idx == TILE_SIZE - 1) begin
                        k_idx <= 0;
                        // Collect MAC results
                        for (i = 0; i < NUM_PES && (col_in_tile + i) < TILE_SIZE; i = i + 1) begin
                            acc[row_in_tile][col_in_tile + i] <= acc[row_in_tile][col_in_tile + i] + mac_acc[i];
                            mac_enable[i] <= 1'b0;
                        end

                        if (col_in_tile + NUM_PES >= TILE_SIZE) begin
                            col_in_tile <= 0;
                            if (row_in_tile == TILE_SIZE - 1) begin
                                row_in_tile <= 0;
                                // Check if more k-tiles remain
                                if ((k_tile_idx + 1) * TILE_SIZE < D_K) begin
                                    k_tile_idx <= k_tile_idx + 1;
                                    state <= S_LOAD_Q;
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
                    // Write accumulated scores to output BRAM
                    score_addr <= (tile_row * TILE_SIZE + load_row) * seq_len + 
                                  (tile_col * TILE_SIZE + load_col);
                    score_data <= acc[load_row][load_col];
                    score_wr_en <= 1'b1;

                    if (load_col == TILE_SIZE - 1) begin
                        load_col <= 0;
                        if (load_row == TILE_SIZE - 1) begin
                            score_wr_en <= 1'b0;
                            state <= S_NEXT_TILE;
                        end else begin
                            load_row <= load_row + 1;
                        end
                    end else begin
                        load_col <= load_col + 1;
                    end
                end

                S_NEXT_TILE: begin
                    // Advance to next output tile
                    k_tile_idx <= 0;
                    // Clear accumulators for next tile
                    for (i = 0; i < TILE_SIZE; i = i + 1)
                        for (j = 0; j < TILE_SIZE; j = j + 1)
                            acc[i][j] <= 32'sd0;

                    if (tile_col + 1 < (seq_len + TILE_SIZE - 1) / TILE_SIZE) begin
                        tile_col <= tile_col + 1;
                        state    <= S_LOAD_Q;
                        load_row <= 0;
                        load_col <= 0;
                    end else begin
                        tile_col <= 0;
                        if (tile_row + 1 < (seq_len + TILE_SIZE - 1) / TILE_SIZE) begin
                            tile_row <= tile_row + 1;
                            state    <= S_LOAD_Q;
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

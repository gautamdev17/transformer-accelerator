//============================================================================
// File:    dma_controller.v
// Desc:    DMA Controller for weight loading and activation transfer
//          Manages data movement between DDR (AXI) and on-chip BRAM
//          Supports burst transfers for efficient bandwidth utilization
//============================================================================
`include "../common/defines.v"

module dma_controller #(
    parameter AXI_ADDR_W = 32,
    parameter AXI_DATA_W = 64,
    parameter BURST_LEN  = `DMA_BURST_LEN
)(
    input  wire        clk,
    input  wire        rst_n,

    // Control interface
    input  wire        start_transfer,
    input  wire [31:0] src_addr,       // DDR source address
    input  wire [31:0] dst_addr,       // BRAM destination address
    input  wire [31:0] transfer_len,   // Number of words to transfer
    input  wire        direction,      // 0 = DDR→BRAM, 1 = BRAM→DDR
    output reg         transfer_done,
    output reg         transfer_busy,

    // AXI Master Read Channel (simplified)
    output reg  [AXI_ADDR_W-1:0] axi_ar_addr,
    output reg  [7:0]            axi_ar_len,
    output reg                   axi_ar_valid,
    input  wire                  axi_ar_ready,
    input  wire [AXI_DATA_W-1:0] axi_r_data,
    input  wire                  axi_r_valid,
    input  wire                  axi_r_last,
    output reg                   axi_r_ready,

    // AXI Master Write Channel (simplified)
    output reg  [AXI_ADDR_W-1:0] axi_aw_addr,
    output reg  [7:0]            axi_aw_len,
    output reg                   axi_aw_valid,
    input  wire                  axi_aw_ready,
    output reg  [AXI_DATA_W-1:0] axi_w_data,
    output reg                   axi_w_valid,
    output reg                   axi_w_last,
    input  wire                  axi_w_ready,
    input  wire                  axi_b_valid,
    output reg                   axi_b_ready,

    // BRAM interface
    output reg  [19:0]           bram_addr,
    output reg  [63:0]           bram_wr_data,
    input  wire [63:0]           bram_rd_data,
    output reg                   bram_wr_en,
    output reg                   bram_rd_en
);

    // ---- FSM ----
    reg [3:0] state;
    localparam S_IDLE        = 4'd0;
    localparam S_RD_REQ      = 4'd1;
    localparam S_RD_DATA     = 4'd2;
    localparam S_WR_REQ      = 4'd3;
    localparam S_WR_DATA     = 4'd4;
    localparam S_WR_RESP     = 4'd5;
    localparam S_NEXT_BURST  = 4'd6;
    localparam S_DONE        = 4'd7;

    reg [31:0] remaining;
    reg [31:0] current_src, current_dst;
    reg [7:0]  burst_count;
    reg [7:0]  current_burst_len;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            transfer_done  <= 1'b0;
            transfer_busy  <= 1'b0;
            axi_ar_valid   <= 1'b0;
            axi_r_ready    <= 1'b0;
            axi_aw_valid   <= 1'b0;
            axi_w_valid    <= 1'b0;
            axi_b_ready    <= 1'b0;
            bram_wr_en     <= 1'b0;
            bram_rd_en     <= 1'b0;
        end else begin
            transfer_done <= 1'b0;
            bram_wr_en    <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start_transfer) begin
                        transfer_busy    <= 1'b1;
                        remaining        <= transfer_len;
                        current_src      <= src_addr;
                        current_dst      <= dst_addr;
                        state            <= direction ? S_WR_REQ : S_RD_REQ;
                    end
                end

                // ---- DDR → BRAM (Read from DDR, write to BRAM) ----
                S_RD_REQ: begin
                    current_burst_len <= (remaining > BURST_LEN) ? BURST_LEN - 1 : remaining[7:0] - 1;
                    axi_ar_addr       <= current_src;
                    axi_ar_len        <= (remaining > BURST_LEN) ? BURST_LEN - 1 : remaining[7:0] - 1;
                    axi_ar_valid      <= 1'b1;

                    if (axi_ar_ready) begin
                        axi_ar_valid <= 1'b0;
                        axi_r_ready  <= 1'b1;
                        burst_count  <= 0;
                        state        <= S_RD_DATA;
                    end
                end

                S_RD_DATA: begin
                    if (axi_r_valid) begin
                        // Write received data to BRAM
                        bram_addr    <= current_dst[19:0] + burst_count;
                        bram_wr_data <= axi_r_data;
                        bram_wr_en   <= 1'b1;
                        burst_count  <= burst_count + 1;

                        if (axi_r_last) begin
                            axi_r_ready <= 1'b0;
                            remaining   <= remaining - (current_burst_len + 1);
                            current_src <= current_src + ((current_burst_len + 1) << 3);
                            current_dst <= current_dst + (current_burst_len + 1);
                            state       <= S_NEXT_BURST;
                        end
                    end
                end

                // ---- BRAM → DDR (Read from BRAM, write to DDR) ----
                S_WR_REQ: begin
                    current_burst_len <= (remaining > BURST_LEN) ? BURST_LEN - 1 : remaining[7:0] - 1;
                    axi_aw_addr       <= current_dst;
                    axi_aw_len        <= (remaining > BURST_LEN) ? BURST_LEN - 1 : remaining[7:0] - 1;
                    axi_aw_valid      <= 1'b1;

                    if (axi_aw_ready) begin
                        axi_aw_valid <= 1'b0;
                        burst_count  <= 0;
                        bram_rd_en   <= 1'b1;
                        bram_addr    <= current_src[19:0];
                        state        <= S_WR_DATA;
                    end
                end

                S_WR_DATA: begin
                    axi_w_data  <= bram_rd_data;
                    axi_w_valid <= 1'b1;
                    axi_w_last  <= (burst_count == current_burst_len);

                    if (axi_w_ready) begin
                        burst_count <= burst_count + 1;
                        bram_addr   <= current_src[19:0] + burst_count + 1;

                        if (burst_count == current_burst_len) begin
                            axi_w_valid <= 1'b0;
                            bram_rd_en  <= 1'b0;
                            axi_b_ready <= 1'b1;
                            state       <= S_WR_RESP;
                        end
                    end
                end

                S_WR_RESP: begin
                    if (axi_b_valid) begin
                        axi_b_ready <= 1'b0;
                        remaining   <= remaining - (current_burst_len + 1);
                        current_src <= current_src + (current_burst_len + 1);
                        current_dst <= current_dst + ((current_burst_len + 1) << 3);
                        state       <= S_NEXT_BURST;
                    end
                end

                S_NEXT_BURST: begin
                    if (remaining == 0) begin
                        state <= S_DONE;
                    end else begin
                        state <= direction ? S_WR_REQ : S_RD_REQ;
                    end
                end

                S_DONE: begin
                    transfer_done <= 1'b1;
                    transfer_busy <= 1'b0;
                    state         <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

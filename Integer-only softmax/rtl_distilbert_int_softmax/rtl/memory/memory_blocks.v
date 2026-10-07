//============================================================================
// File:    weight_bram.v
// Desc:    Weight storage BRAM — stores model weights for all layers
//          Dual-port: Port A for DMA loading, Port B for compute read
//          Total capacity: sized for DistilBERT (~67 MB quantized)
//============================================================================

module weight_bram #(
    parameter DATA_WIDTH = 64,    // 8 x INT8 per word
    parameter ADDR_WIDTH = 20,    // 2^20 = 1M words = 8 MB per bank
    parameter DEPTH      = 1 << ADDR_WIDTH
)(
    input  wire                    clk,

    // Port A: DMA write
    input  wire                    a_wr_en,
    input  wire [ADDR_WIDTH-1:0]   a_addr,
    input  wire [DATA_WIDTH-1:0]   a_wr_data,

    // Port B: Compute read
    input  wire                    b_rd_en,
    input  wire [ADDR_WIDTH-1:0]   b_addr,
    output reg  [DATA_WIDTH-1:0]   b_rd_data
);

    (* ram_style = "block" *)
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    // Port A: Write
    always @(posedge clk) begin
        if (a_wr_en)
            mem[a_addr] <= a_wr_data;
    end

    // Port B: Read
    always @(posedge clk) begin
        if (b_rd_en)
            b_rd_data <= mem[b_addr];
    end

endmodule

//============================================================================
// File:    activation_bram.v
// Desc:    Activation buffer BRAM — double-buffered for pipeline overlap
//          Stores intermediate activations between transformer stages
//============================================================================
module activation_bram #(
    parameter DATA_WIDTH = 64,
    parameter ADDR_WIDTH = 18,    // 2^18 = 256K words
    parameter DEPTH      = 1 << ADDR_WIDTH
)(
    input  wire                    clk,

    // Port A: Write (from compute)
    input  wire                    a_wr_en,
    input  wire [ADDR_WIDTH-1:0]   a_addr,
    input  wire [DATA_WIDTH-1:0]   a_wr_data,

    // Port B: Read (to compute)
    input  wire                    b_rd_en,
    input  wire [ADDR_WIDTH-1:0]   b_addr,
    output reg  [DATA_WIDTH-1:0]   b_rd_data
);

    (* ram_style = "block" *)
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (a_wr_en)
            mem[a_addr] <= a_wr_data;
    end

    always @(posedge clk) begin
        if (b_rd_en)
            b_rd_data <= mem[b_addr];
    end

endmodule

//============================================================================
// File:    embedding_rom.v
// Desc:    Embedding lookup table — stores token embeddings
//          Vocab: 30522 × 768 = ~23.4M INT8 values
//          Loaded from DDR at initialization via DMA
//============================================================================
module embedding_rom #(
    parameter VOCAB_SIZE  = `VOCAB_SIZE,
    parameter D_MODEL     = `D_MODEL,
    parameter ADDR_WIDTH  = 15       // Covers 30522 entries
)(
    input  wire                    clk,

    // Lookup interface
    input  wire                    rd_en,
    input  wire [ADDR_WIDTH-1:0]   token_id,
    input  wire [9:0]              dim_idx,
    output reg  signed [7:0]       embed_out,

    // Load interface (from DMA)
    input  wire                    wr_en,
    input  wire [ADDR_WIDTH-1:0]   wr_token_id,
    input  wire [9:0]              wr_dim_idx,
    input  wire signed [7:0]       wr_data
);

    // Embedding storage (block RAM)
    // Practical note: full 30522×768 won't fit in BRAM;
    // use external DDR + caching in real implementation
    (* ram_style = "block" *)
    reg signed [7:0] embed_mem [0:8191][0:767]; // Subset for BRAM

    always @(posedge clk) begin
        if (wr_en)
            embed_mem[wr_token_id[12:0]][wr_dim_idx] <= wr_data;
    end

    always @(posedge clk) begin
        if (rd_en)
            embed_out <= embed_mem[token_id[12:0]][dim_idx];
    end

endmodule

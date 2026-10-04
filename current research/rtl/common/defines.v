//============================================================================
// File:    defines.v
// Project: MPTRANS - Mixed-Precision DistilBERT Accelerator
// Desc:    Global parameters and defines for the DistilBERT inference engine
//============================================================================

`ifndef DEFINES_V
`define DEFINES_V

// ======================== Model Architecture ================================
`define D_MODEL         768     // Hidden dimension
`define N_HEADS         12      // Number of attention heads
`define D_K             64      // Per-head dimension (D_MODEL / N_HEADS)
`define D_FF            3072    // Feed-forward intermediate dimension (4 * D_MODEL)
`define N_LAYERS        6       // Number of transformer layers
`define VOCAB_SIZE      30522   // WordPiece vocabulary size
`define NUM_CLASSES     18      // Amazon MASSIVE intent classes
`define MAX_SEQ_LEN     128     // Maximum sequence length supported

// ======================== Precision Configuration ===========================
// Mixed-precision: INT8 for QK, FP16 for Softmax, INT4 for AV
`define INT8_WIDTH      8
`define INT4_WIDTH      4
`define FP16_WIDTH      16
`define INT16_WIDTH     16
`define INT32_WIDTH     32
`define ACC_WIDTH        32     // Accumulator width for MAC operations

// FP16 format: 1 sign + 5 exponent + 10 mantissa
`define FP16_SIGN_BIT   15
`define FP16_EXP_MSB    14
`define FP16_EXP_LSB    10
`define FP16_EXP_WIDTH  5
`define FP16_MAN_MSB    9
`define FP16_MAN_LSB    0
`define FP16_MAN_WIDTH  10
`define FP16_EXP_BIAS   15

// Fixed-point Q8.8 format for GELU
`define FIXED_WIDTH     16
`define FIXED_FRAC      8

// ======================== Tiling Configuration ==============================
`define TILE_SIZE       8       // Tile dimension for tiled matrix multiplication
`define NUM_PES         8       // Number of processing elements (parallel MACs)

// ======================== Memory Configuration ==============================
`define WEIGHT_ADDR_W   20      // Weight BRAM address width
`define ACT_ADDR_W      18      // Activation BRAM address width
`define BRAM_DATA_W     64      // BRAM data bus width (8 x INT8)
`define EMBED_ADDR_W    15      // Embedding ROM address width
`define DMA_BURST_LEN   16      // DMA burst length

// ======================== FSM States — Main Controller ======================
`define FSM_IDLE            4'd0
`define FSM_LOAD_TOKENS     4'd1
`define FSM_EMBEDDING       4'd2
`define FSM_LAYER_NORM1     4'd3
`define FSM_ATTENTION       4'd4
`define FSM_RESIDUAL1       4'd5
`define FSM_LAYER_NORM2     4'd6
`define FSM_FFN             4'd7
`define FSM_RESIDUAL2       4'd8
`define FSM_NEXT_LAYER      4'd9
`define FSM_CLASSIFY        4'd10
`define FSM_DONE            4'd11

// ======================== FSM States — Attention ============================
`define ATT_IDLE            4'd0
`define ATT_PROJ_QKV        4'd1
`define ATT_QK_MATMUL       4'd2
`define ATT_SCALE           4'd3
`define ATT_SOFTMAX         4'd4
`define ATT_AV_MATMUL       4'd5
`define ATT_CONCAT          4'd6
`define ATT_OUT_PROJ        4'd7
`define ATT_DONE            4'd8

// ======================== FSM States — FFN ==================================
`define FFN_IDLE            3'd0
`define FFN_LINEAR1         3'd1
`define FFN_GELU            3'd2
`define FFN_LINEAR2         3'd3
`define FFN_DONE            3'd4

// ======================== Custom NLP ISA Opcodes ============================
`define NLP_OPCODE          7'b0001011   // custom-0 opcode space
`define NLP_FUNCT3_LOAD     3'b000       // nlp.load  — load token IDs
`define NLP_FUNCT3_RUN      3'b001       // nlp.run   — start inference
`define NLP_FUNCT3_BUSY     3'b010       // nlp.busy  — poll status
`define NLP_FUNCT3_RESULT   3'b011       // nlp.result — read class output
`define NLP_FUNCT3_CONFIG   3'b100       // nlp.config — set seq_len etc.

`endif // DEFINES_V

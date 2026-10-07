//============================================================================
// File:    transformer_fsm.v
// Desc:    Main FSM Controller for DistilBERT inference pipeline
//          Orchestrates: Embedding → 6×(LN → MHA → Residual → LN → FFN → Residual) → Classify
//          Time-multiplexes hardware across all 6 layers
//============================================================================
`include "../common/defines.v"

module transformer_fsm #(
    parameter N_LAYERS  = `N_LAYERS,
    parameter D_MODEL   = `D_MODEL,
    parameter NUM_CLASSES = `NUM_CLASSES
)(
    input  wire        clk,
    input  wire        rst_n,

    // Control interface (from RISC-V custom ISA decoder)
    input  wire        nlp_start,       // Start inference
    input  wire        nlp_load,        // Load token IDs
    input  wire [15:0] nlp_token_data,  // Token ID input
    input  wire [6:0]  nlp_seq_len,     // Sequence length
    output reg         nlp_busy,        // Inference in progress
    output reg         nlp_done,        // Inference complete
    output reg  [4:0]  nlp_class_out,   // Predicted class (0-17)
    output reg  [31:0] nlp_logit_out,   // Top logit value

    // Sub-module control signals
    output reg         ln1_start,
    input  wire        ln1_done,
    output reg         mha_start,
    input  wire        mha_done,
    output reg         ln2_start,
    input  wire        ln2_done,
    output reg         ffn_start,
    input  wire        ffn_done,

    // Embedding lookup
    output reg         embed_start,
    input  wire        embed_done,

    // Classification head
    output reg         cls_start,
    input  wire        cls_done,
    input  wire [4:0]  cls_class,
    input  wire [31:0] cls_logit,

    // Residual add control
    output reg         residual_start,
    output reg         residual_sel,    // 0 = post-MHA, 1 = post-FFN
    input  wire        residual_done,

    // Sequence length propagation
    output reg  [6:0]  seq_len_out,

    // Layer index (for weight address offset)
    output reg  [2:0]  current_layer
);

    // ---- Main FSM ----
    reg [3:0] state;

    // Token buffer
    reg [15:0] token_ids [0:`MAX_SEQ_LEN-1];
    reg [6:0]  token_count;
    reg [6:0]  seq_len_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= `FSM_IDLE;
            nlp_busy       <= 1'b0;
            nlp_done       <= 1'b0;
            nlp_class_out  <= 5'd0;
            nlp_logit_out  <= 32'd0;
            current_layer  <= 3'd0;
            token_count    <= 7'd0;
            seq_len_reg    <= 7'd0;
            ln1_start      <= 1'b0;
            mha_start      <= 1'b0;
            ln2_start      <= 1'b0;
            ffn_start      <= 1'b0;
            embed_start    <= 1'b0;
            cls_start      <= 1'b0;
            residual_start <= 1'b0;
        end else begin
            // Default pulse signals
            ln1_start      <= 1'b0;
            mha_start      <= 1'b0;
            ln2_start      <= 1'b0;
            ffn_start      <= 1'b0;
            embed_start    <= 1'b0;
            cls_start      <= 1'b0;
            residual_start <= 1'b0;
            nlp_done       <= 1'b0;

            case (state)
                `FSM_IDLE: begin
                    // Accept token loads
                    if (nlp_load) begin
                        token_ids[token_count] <= nlp_token_data;
                        token_count <= token_count + 1;
                    end

                    if (nlp_start) begin
                        nlp_busy      <= 1'b1;
                        seq_len_reg   <= (nlp_seq_len != 0) ? nlp_seq_len : token_count;
                        seq_len_out   <= (nlp_seq_len != 0) ? nlp_seq_len : token_count;
                        state         <= `FSM_LOAD_TOKENS;
                    end
                end

                `FSM_LOAD_TOKENS: begin
                    // Tokens already loaded, proceed to embedding
                    state       <= `FSM_EMBEDDING;
                    embed_start <= 1'b1;
                end

                `FSM_EMBEDDING: begin
                    if (embed_done) begin
                        current_layer <= 3'd0;
                        state         <= `FSM_LAYER_NORM1;
                        ln1_start     <= 1'b1;
                    end
                end

                `FSM_LAYER_NORM1: begin
                    if (ln1_done) begin
                        state     <= `FSM_ATTENTION;
                        mha_start <= 1'b1;
                    end
                end

                `FSM_ATTENTION: begin
                    if (mha_done) begin
                        state          <= `FSM_RESIDUAL1;
                        residual_start <= 1'b1;
                        residual_sel   <= 1'b0;  // Post-MHA residual
                    end
                end

                `FSM_RESIDUAL1: begin
                    if (residual_done) begin
                        state     <= `FSM_LAYER_NORM2;
                        ln2_start <= 1'b1;
                    end
                end

                `FSM_LAYER_NORM2: begin
                    if (ln2_done) begin
                        state     <= `FSM_FFN;
                        ffn_start <= 1'b1;
                    end
                end

                `FSM_FFN: begin
                    if (ffn_done) begin
                        state          <= `FSM_RESIDUAL2;
                        residual_start <= 1'b1;
                        residual_sel   <= 1'b1;  // Post-FFN residual
                    end
                end

                `FSM_RESIDUAL2: begin
                    if (residual_done) begin
                        state <= `FSM_NEXT_LAYER;
                    end
                end

                `FSM_NEXT_LAYER: begin
                    if (current_layer == N_LAYERS - 1) begin
                        // All 6 layers done → classify
                        state     <= `FSM_CLASSIFY;
                        cls_start <= 1'b1;
                    end else begin
                        current_layer <= current_layer + 1;
                        state         <= `FSM_LAYER_NORM1;
                        ln1_start     <= 1'b1;
                    end
                end

                `FSM_CLASSIFY: begin
                    if (cls_done) begin
                        nlp_class_out <= cls_class;
                        nlp_logit_out <= cls_logit;
                        state         <= `FSM_DONE;
                    end
                end

                `FSM_DONE: begin
                    nlp_done    <= 1'b1;
                    nlp_busy    <= 1'b0;
                    token_count <= 7'd0;
                    state       <= `FSM_IDLE;
                end

                default: state <= `FSM_IDLE;
            endcase
        end
    end

endmodule

//============================================================================
// File:    classifier_head.v
// Desc:    Classification Head for Amazon MASSIVE intent classification
//          1) Mean pooling over sequence dimension
//          2) Linear projection: [d_model] → [num_classes]
//          3) Argmax to find predicted class
//============================================================================
`include "../common/defines.v"

module classifier_head #(
    parameter D_MODEL     = `D_MODEL,
    parameter NUM_CLASSES = `NUM_CLASSES
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [6:0]  seq_len,

    // Input: final hidden states (seq_len × d_model, INT8)
    output reg  [17:0] input_addr,
    input  wire signed [7:0] input_data,
    output reg         input_rd_en,

    // Classifier weights (d_model × num_classes, INT8)
    output reg  [13:0] cls_w_addr,
    input  wire signed [7:0] cls_w_data,
    output reg         cls_w_rd_en,

    // Classifier bias (num_classes, INT32)
    output reg  [4:0]  cls_b_addr,
    input  wire signed [31:0] cls_b_data,
    output reg         cls_b_rd_en,

    // Output
    output reg  [4:0]  predicted_class,
    output reg  [31:0] top_logit,
    output reg         done,
    output reg         busy
);

    // FSM
    reg [2:0] state;
    localparam S_IDLE    = 3'd0;
    localparam S_POOL    = 3'd1;
    localparam S_LINEAR  = 3'd2;
    localparam S_ARGMAX  = 3'd3;
    localparam S_DONE    = 3'd4;

    // Pooled vector: mean over sequence (INT32 accumulation → INT8)
    reg signed [31:0] pool_acc [0:D_MODEL-1];
    reg signed [7:0]  pooled [0:D_MODEL-1];

    // Logits
    reg signed [31:0] logits [0:NUM_CLASSES-1];

    // Counters
    reg [6:0]  seq_idx;
    reg [9:0]  dim_idx;
    reg [4:0]  cls_idx;
    reg signed [31:0] acc;

    // Argmax
    reg signed [31:0] max_val;
    reg [4:0]  max_idx;

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            done           <= 1'b0;
            busy           <= 1'b0;
            input_rd_en    <= 1'b0;
            cls_w_rd_en    <= 1'b0;
            cls_b_rd_en    <= 1'b0;
            predicted_class <= 5'd0;
            top_logit      <= 32'd0;
        end else begin
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy    <= 1'b1;
                        seq_idx <= 0;
                        dim_idx <= 0;
                        for (i = 0; i < D_MODEL; i = i + 1)
                            pool_acc[i] <= 32'sd0;
                        state <= S_POOL;
                    end
                end

                S_POOL: begin
                    // Mean pooling: pool[d] = sum(x[s][d]) / seq_len for all s
                    input_addr  <= seq_idx * D_MODEL + dim_idx;
                    input_rd_en <= 1'b1;

                    if (input_rd_en) begin
                        pool_acc[dim_idx] <= pool_acc[dim_idx] + {{24{input_data[7]}}, input_data};

                        if (dim_idx == D_MODEL - 1) begin
                            dim_idx <= 0;
                            if (seq_idx == seq_len - 1) begin
                                // Compute mean and quantize
                                for (i = 0; i < D_MODEL; i = i + 1) begin
                                    // Divide by seq_len (approximate)
                                    pooled[i] <= pool_acc[i] / {{25{seq_len[6]}}, seq_len};
                                end
                                seq_idx     <= 0;
                                cls_idx     <= 0;
                                dim_idx     <= 0;
                                acc         <= 0;
                                input_rd_en <= 1'b0;
                                state       <= S_LINEAR;
                            end else begin
                                seq_idx <= seq_idx + 1;
                            end
                        end else begin
                            dim_idx <= dim_idx + 1;
                        end
                    end
                end

                S_LINEAR: begin
                    // Linear: logit[c] = sum(pooled[d] * W[d][c]) + bias[c]
                    cls_w_addr  <= dim_idx * NUM_CLASSES + cls_idx;
                    cls_w_rd_en <= 1'b1;

                    if (cls_w_rd_en) begin
                        acc <= acc + $signed(pooled[dim_idx]) * $signed(cls_w_data);

                        if (dim_idx == D_MODEL - 1) begin
                            dim_idx <= 0;
                            // Add bias
                            cls_b_addr  <= cls_idx;
                            cls_b_rd_en <= 1'b1;
                            logits[cls_idx] <= acc + cls_b_data;
                            acc <= 0;

                            if (cls_idx == NUM_CLASSES - 1) begin
                                cls_w_rd_en <= 1'b0;
                                cls_b_rd_en <= 1'b0;
                                state       <= S_ARGMAX;
                                max_val     <= 32'sh80000000;
                                max_idx     <= 0;
                                cls_idx     <= 0;
                            end else begin
                                cls_idx <= cls_idx + 1;
                            end
                        end else begin
                            dim_idx <= dim_idx + 1;
                        end
                    end
                end

                S_ARGMAX: begin
                    // Find class with highest logit
                    if ($signed(logits[cls_idx]) > $signed(max_val)) begin
                        max_val <= logits[cls_idx];
                        max_idx <= cls_idx;
                    end

                    if (cls_idx == NUM_CLASSES - 1) begin
                        predicted_class <= max_idx;
                        top_logit       <= max_val;
                        state           <= S_DONE;
                    end else begin
                        cls_idx <= cls_idx + 1;
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

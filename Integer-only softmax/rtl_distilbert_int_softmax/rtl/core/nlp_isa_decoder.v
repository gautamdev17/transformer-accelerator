//============================================================================
// File:    nlp_isa_decoder.v
// Desc:    Custom NLP ISA Extension Decoder for RISC-V
//          Decodes custom-0 opcode (0001011) with NLP-specific funct3 fields
//          Instructions:
//            nlp.load  rs1, rs2   — Load token ID rs2 at position rs1
//            nlp.run   rs1        — Start inference, rs1 = seq_len
//            nlp.busy  rd         — Read busy flag into rd
//            nlp.result rd        — Read classification result into rd
//            nlp.config rs1, rs2  — Configure accelerator parameters
//============================================================================
`include "../common/defines.v"

module nlp_isa_decoder (
    input  wire        clk,
    input  wire        rst_n,

    // Instruction interface (from RISC-V decode stage)
    input  wire [31:0] instruction,
    input  wire        instr_valid,
    input  wire [31:0] rs1_data,
    input  wire [31:0] rs2_data,

    // Writeback interface (to RISC-V)
    output reg  [31:0] rd_data,
    output reg         rd_write_en,
    output reg         stall,          // Stall pipeline during accelerator ops

    // Accelerator control interface
    output reg         nlp_load,
    output reg  [15:0] nlp_token_data,
    output reg         nlp_start,
    output reg  [6:0]  nlp_seq_len,
    input  wire        nlp_busy,
    input  wire        nlp_done,
    input  wire [4:0]  nlp_class_out,
    input  wire [31:0] nlp_logit_out,

    // Custom instruction detected flag
    output wire        is_nlp_instr
);

    // Decode fields
    wire [6:0] opcode  = instruction[6:0];
    wire [2:0] funct3  = instruction[14:12];
    wire [4:0] rd      = instruction[11:7];
    wire [4:0] rs1     = instruction[19:15];
    wire [4:0] rs2     = instruction[24:20];

    assign is_nlp_instr = (opcode == `NLP_OPCODE) && instr_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nlp_load       <= 1'b0;
            nlp_start      <= 1'b0;
            nlp_token_data <= 16'd0;
            nlp_seq_len    <= 7'd0;
            rd_data        <= 32'd0;
            rd_write_en    <= 1'b0;
            stall          <= 1'b0;
        end else begin
            // Default deasserts
            nlp_load    <= 1'b0;
            nlp_start   <= 1'b0;
            rd_write_en <= 1'b0;
            stall       <= 1'b0;

            if (is_nlp_instr) begin
                case (funct3)
                    `NLP_FUNCT3_LOAD: begin
                        // nlp.load: token_data = rs2_data, position = rs1_data
                        nlp_load       <= 1'b1;
                        nlp_token_data <= rs2_data[15:0];
                    end

                    `NLP_FUNCT3_RUN: begin
                        // nlp.run: start inference, seq_len = rs1_data
                        nlp_start   <= 1'b1;
                        nlp_seq_len <= rs1_data[6:0];
                    end

                    `NLP_FUNCT3_BUSY: begin
                        // nlp.busy: rd = {busy, done}
                        rd_data     <= {30'd0, nlp_done, nlp_busy};
                        rd_write_en <= 1'b1;
                    end

                    `NLP_FUNCT3_RESULT: begin
                        // nlp.result: rd = {logit[31:5], class[4:0]}
                        rd_data     <= {nlp_logit_out[31:5], nlp_class_out};
                        rd_write_en <= 1'b1;
                    end

                    `NLP_FUNCT3_CONFIG: begin
                        // nlp.config: rs1 = config_type, rs2 = config_value
                        // Reserved for future: precision mode, tile size, etc.
                        // Currently no-op, extensibility point
                    end

                    default: ;
                endcase

                // Stall core while accelerator is busy
                if (funct3 == `NLP_FUNCT3_RUN)
                    stall <= 1'b1;
            end
        end
    end

endmodule

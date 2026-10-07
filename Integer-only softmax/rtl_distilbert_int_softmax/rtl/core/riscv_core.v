//============================================================================
// File:    riscv_core.v
// Desc:    Simplified RISC-V RV32I scalar core with NLP ISA extension
//          5-stage pipeline: IF → ID → EX → MEM → WB
//          Supports RV32I base + custom-0 NLP instructions
//============================================================================
`include "../common/defines.v"

module riscv_core (
    input  wire        clk,
    input  wire        rst_n,

    // Instruction memory interface
    output reg  [31:0] imem_addr,
    input  wire [31:0] imem_data,
    output reg         imem_rd_en,

    // Data memory interface
    output reg  [31:0] dmem_addr,
    input  wire [31:0] dmem_rd_data,
    output reg  [31:0] dmem_wr_data,
    output reg         dmem_rd_en,
    output reg         dmem_wr_en,

    // NLP Accelerator interface
    output wire        nlp_load,
    output wire [15:0] nlp_token_data,
    output wire        nlp_start,
    output wire [6:0]  nlp_seq_len,
    input  wire        nlp_busy,
    input  wire        nlp_done,
    input  wire [4:0]  nlp_class_out,
    input  wire [31:0] nlp_logit_out,

    // External interrupt
    input  wire        ext_irq,
    output reg         halted
);

    // ---- Register File ----
    reg signed [31:0] regfile [0:31];

    // ---- Pipeline Registers ----
    // IF/ID
    reg [31:0] if_id_pc, if_id_instr;
    reg        if_id_valid;

    // ID/EX
    reg [31:0] id_ex_pc, id_ex_rs1, id_ex_rs2, id_ex_imm;
    reg [4:0]  id_ex_rd, id_ex_rs1_addr, id_ex_rs2_addr;
    reg [6:0]  id_ex_opcode;
    reg [2:0]  id_ex_funct3;
    reg [6:0]  id_ex_funct7;
    reg        id_ex_valid;

    // EX/MEM
    reg [31:0] ex_mem_result, ex_mem_rs2;
    reg [4:0]  ex_mem_rd;
    reg        ex_mem_mem_read, ex_mem_mem_write, ex_mem_reg_write;
    reg        ex_mem_valid;

    // MEM/WB
    reg [31:0] mem_wb_result;
    reg [4:0]  mem_wb_rd;
    reg        mem_wb_reg_write;
    reg        mem_wb_valid;

    // Program Counter
    reg [31:0] pc;
    wire       pipeline_stall;
    wire       nlp_stall;

    // ---- NLP ISA Decoder ----
    wire [31:0] nlp_rd_data;
    wire        nlp_rd_write;
    wire        is_nlp;

    nlp_isa_decoder u_nlp_dec (
        .clk(clk), .rst_n(rst_n),
        .instruction(if_id_instr),
        .instr_valid(if_id_valid),
        .rs1_data(regfile[if_id_instr[19:15]]),
        .rs2_data(regfile[if_id_instr[24:20]]),
        .rd_data(nlp_rd_data),
        .rd_write_en(nlp_rd_write),
        .stall(nlp_stall),
        .nlp_load(nlp_load),
        .nlp_token_data(nlp_token_data),
        .nlp_start(nlp_start),
        .nlp_seq_len(nlp_seq_len),
        .nlp_busy(nlp_busy),
        .nlp_done(nlp_done),
        .nlp_class_out(nlp_class_out),
        .nlp_logit_out(nlp_logit_out),
        .is_nlp_instr(is_nlp)
    );

    assign pipeline_stall = nlp_stall || nlp_busy;

    // ---- Immediate Extraction ----
    function [31:0] extract_imm;
        input [31:0] instr;
        input [6:0]  opcode;
        begin
            case (opcode)
                7'b0010011, // I-type (ADDI, etc.)
                7'b0000011: // Load
                    extract_imm = {{20{instr[31]}}, instr[31:20]};
                7'b0100011: // S-type (Store)
                    extract_imm = {{20{instr[31]}}, instr[31:25], instr[11:7]};
                7'b1100011: // B-type (Branch)
                    extract_imm = {{19{instr[31]}}, instr[31], instr[7], instr[30:25], instr[11:8], 1'b0};
                7'b0110111, // U-type (LUI)
                7'b0010111: // AUIPC
                    extract_imm = {instr[31:12], 12'b0};
                7'b1101111: // J-type (JAL)
                    extract_imm = {{11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};
                default:
                    extract_imm = 32'd0;
            endcase
        end
    endfunction

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc            <= 32'h0000_0000;
            if_id_valid   <= 1'b0;
            id_ex_valid   <= 1'b0;
            ex_mem_valid  <= 1'b0;
            mem_wb_valid  <= 1'b0;
            halted        <= 1'b0;
            imem_rd_en    <= 1'b0;
            dmem_rd_en    <= 1'b0;
            dmem_wr_en    <= 1'b0;
            for (i = 0; i < 32; i = i + 1)
                regfile[i] <= 32'd0;
        end else if (!pipeline_stall) begin

            // ========== Stage 1: FETCH ==========
            imem_addr  <= pc;
            imem_rd_en <= 1'b1;
            if_id_pc   <= pc;
            if_id_instr <= imem_data;
            if_id_valid <= 1'b1;
            pc         <= pc + 4;

            // ========== Stage 2: DECODE ==========
            if (if_id_valid) begin
                id_ex_opcode    <= if_id_instr[6:0];
                id_ex_funct3    <= if_id_instr[14:12];
                id_ex_funct7    <= if_id_instr[31:25];
                id_ex_rd        <= if_id_instr[11:7];
                id_ex_rs1_addr  <= if_id_instr[19:15];
                id_ex_rs2_addr  <= if_id_instr[24:20];
                id_ex_rs1       <= regfile[if_id_instr[19:15]];
                id_ex_rs2       <= regfile[if_id_instr[24:20]];
                id_ex_imm       <= extract_imm(if_id_instr, if_id_instr[6:0]);
                id_ex_pc        <= if_id_pc;
                id_ex_valid     <= 1'b1;

                // Handle NLP custom instructions (bypass normal pipeline)
                if (is_nlp && nlp_rd_write) begin
                    regfile[if_id_instr[11:7]] <= nlp_rd_data;
                end
            end else begin
                id_ex_valid <= 1'b0;
            end

            // ========== Stage 3: EXECUTE ==========
            if (id_ex_valid) begin
                ex_mem_rd        <= id_ex_rd;
                ex_mem_rs2       <= id_ex_rs2;
                ex_mem_mem_read  <= 1'b0;
                ex_mem_mem_write <= 1'b0;
                ex_mem_reg_write <= 1'b0;
                ex_mem_valid     <= 1'b1;

                case (id_ex_opcode)
                    7'b0110011: begin // R-type
                        ex_mem_reg_write <= 1'b1;
                        case ({id_ex_funct7, id_ex_funct3})
                            10'b0000000_000: ex_mem_result <= id_ex_rs1 + id_ex_rs2;   // ADD
                            10'b0100000_000: ex_mem_result <= id_ex_rs1 - id_ex_rs2;   // SUB
                            10'b0000000_001: ex_mem_result <= id_ex_rs1 << id_ex_rs2[4:0]; // SLL
                            10'b0000000_010: ex_mem_result <= ($signed(id_ex_rs1) < $signed(id_ex_rs2)); // SLT
                            10'b0000000_100: ex_mem_result <= id_ex_rs1 ^ id_ex_rs2;   // XOR
                            10'b0000000_101: ex_mem_result <= id_ex_rs1 >> id_ex_rs2[4:0]; // SRL
                            10'b0000000_110: ex_mem_result <= id_ex_rs1 | id_ex_rs2;   // OR
                            10'b0000000_111: ex_mem_result <= id_ex_rs1 & id_ex_rs2;   // AND
                            default:         ex_mem_result <= 32'd0;
                        endcase
                    end

                    7'b0010011: begin // I-type ALU
                        ex_mem_reg_write <= 1'b1;
                        case (id_ex_funct3)
                            3'b000: ex_mem_result <= id_ex_rs1 + id_ex_imm;            // ADDI
                            3'b010: ex_mem_result <= ($signed(id_ex_rs1) < $signed(id_ex_imm)); // SLTI
                            3'b100: ex_mem_result <= id_ex_rs1 ^ id_ex_imm;            // XORI
                            3'b110: ex_mem_result <= id_ex_rs1 | id_ex_imm;            // ORI
                            3'b111: ex_mem_result <= id_ex_rs1 & id_ex_imm;            // ANDI
                            3'b001: ex_mem_result <= id_ex_rs1 << id_ex_imm[4:0];      // SLLI
                            3'b101: ex_mem_result <= id_ex_rs1 >> id_ex_imm[4:0];      // SRLI
                            default: ex_mem_result <= 32'd0;
                        endcase
                    end

                    7'b0000011: begin // Load
                        ex_mem_result    <= id_ex_rs1 + id_ex_imm;
                        ex_mem_mem_read  <= 1'b1;
                        ex_mem_reg_write <= 1'b1;
                    end

                    7'b0100011: begin // Store
                        ex_mem_result    <= id_ex_rs1 + id_ex_imm;
                        ex_mem_mem_write <= 1'b1;
                    end

                    7'b0110111: begin // LUI
                        ex_mem_result    <= id_ex_imm;
                        ex_mem_reg_write <= 1'b1;
                    end

                    7'b1101111: begin // JAL
                        ex_mem_result    <= id_ex_pc + 4;
                        ex_mem_reg_write <= 1'b1;
                        pc               <= id_ex_pc + id_ex_imm;
                    end

                    7'b1100111: begin // JALR
                        ex_mem_result    <= id_ex_pc + 4;
                        ex_mem_reg_write <= 1'b1;
                        pc               <= (id_ex_rs1 + id_ex_imm) & ~32'd1;
                    end

                    7'b1100011: begin // Branch
                        ex_mem_valid <= 1'b0;
                        case (id_ex_funct3)
                            3'b000: if (id_ex_rs1 == id_ex_rs2) pc <= id_ex_pc + id_ex_imm; // BEQ
                            3'b001: if (id_ex_rs1 != id_ex_rs2) pc <= id_ex_pc + id_ex_imm; // BNE
                            3'b100: if ($signed(id_ex_rs1) < $signed(id_ex_rs2)) pc <= id_ex_pc + id_ex_imm; // BLT
                            3'b101: if ($signed(id_ex_rs1) >= $signed(id_ex_rs2)) pc <= id_ex_pc + id_ex_imm; // BGE
                            default: ;
                        endcase
                    end

                    7'b1110011: begin // ECALL/EBREAK → halt
                        halted <= 1'b1;
                    end

                    default: ex_mem_valid <= 1'b0;
                endcase
            end else begin
                ex_mem_valid <= 1'b0;
            end

            // ========== Stage 4: MEMORY ==========
            dmem_rd_en <= 1'b0;
            dmem_wr_en <= 1'b0;

            if (ex_mem_valid) begin
                mem_wb_rd        <= ex_mem_rd;
                mem_wb_reg_write <= ex_mem_reg_write;
                mem_wb_valid     <= 1'b1;

                if (ex_mem_mem_read) begin
                    dmem_addr   <= ex_mem_result;
                    dmem_rd_en  <= 1'b1;
                    mem_wb_result <= dmem_rd_data;
                end else if (ex_mem_mem_write) begin
                    dmem_addr    <= ex_mem_result;
                    dmem_wr_data <= ex_mem_rs2;
                    dmem_wr_en   <= 1'b1;
                    mem_wb_valid <= 1'b0;
                end else begin
                    mem_wb_result <= ex_mem_result;
                end
            end else begin
                mem_wb_valid <= 1'b0;
            end

            // ========== Stage 5: WRITEBACK ==========
            if (mem_wb_valid && mem_wb_reg_write && mem_wb_rd != 5'd0) begin
                regfile[mem_wb_rd] <= mem_wb_result;
            end

        end // !pipeline_stall

        // x0 is always zero
        regfile[0] <= 32'd0;
    end

endmodule

//============================================================================
// File:    riscv_nlp_soc.v
// Desc:    Complete SoC — RISC-V Core + DistilBERT Accelerator
//          Target: Xilinx ZCU104 (XCZU7EV-2FFVC1156)
//          Integrates: RV32I CPU, NLP Accelerator, memories, AXI bus
//============================================================================
`include "../common/defines.v"

module riscv_nlp_soc #(
    parameter IMEM_SIZE = 16384,   // 16K instructions
    parameter DMEM_SIZE = 65536    // 64K data words
)(
    input  wire        clk,
    input  wire        rst_n,

    // External AXI4 Master (to DDR4 via ZCU104 PS)
    output wire [31:0] axi_ar_addr,
    output wire [7:0]  axi_ar_len,
    output wire        axi_ar_valid,
    input  wire        axi_ar_ready,
    input  wire [63:0] axi_r_data,
    input  wire        axi_r_valid,
    input  wire        axi_r_last,
    output wire        axi_r_ready,

    output wire [31:0] axi_aw_addr,
    output wire [7:0]  axi_aw_len,
    output wire        axi_aw_valid,
    input  wire        axi_aw_ready,
    output wire [63:0] axi_w_data,
    output wire        axi_w_valid,
    output wire        axi_w_last,
    input  wire        axi_w_ready,
    input  wire        axi_b_valid,
    output wire        axi_b_ready,

    // Debug/Status outputs
    output wire        cpu_halted,
    output wire        nlp_busy,
    output wire        nlp_done,
    output wire [4:0]  nlp_class,
    output wire [2:0]  nlp_layer,

    // UART TX for result output (optional)
    output wire        uart_tx
);

    // ---- Instruction Memory ----
    reg [31:0] imem [0:IMEM_SIZE-1];
    wire [31:0] imem_addr;
    wire [31:0] imem_data;
    wire        imem_rd;

    assign imem_data = imem[imem_addr[15:2]]; // Word-aligned

    // Initialize with firmware (in simulation, use $readmemh)
    initial begin
        $readmemh("firmware.hex", imem);
    end

    // ---- Data Memory ----
    reg [31:0] dmem [0:DMEM_SIZE-1];
    wire [31:0] dmem_addr;
    wire [31:0] dmem_rd_data;
    wire [31:0] dmem_wr_data;
    wire        dmem_rd, dmem_wr;

    assign dmem_rd_data = dmem[dmem_addr[17:2]];

    always @(posedge clk) begin
        if (dmem_wr)
            dmem[dmem_addr[17:2]] <= dmem_wr_data;
    end

    // ---- NLP Accelerator interface wires ----
    wire        nlp_load;
    wire [15:0] nlp_token_data;
    wire        nlp_start;
    wire [6:0]  nlp_seq_len;
    wire [31:0] nlp_logit;
    wire [4:0]  nlp_class_out;

    // ---- RISC-V Core ----
    riscv_core u_cpu (
        .clk(clk), .rst_n(rst_n),
        .imem_addr(imem_addr), .imem_data(imem_data), .imem_rd_en(imem_rd),
        .dmem_addr(dmem_addr), .dmem_rd_data(dmem_rd_data),
        .dmem_wr_data(dmem_wr_data), .dmem_rd_en(dmem_rd), .dmem_wr_en(dmem_wr),
        .nlp_load(nlp_load), .nlp_token_data(nlp_token_data),
        .nlp_start(nlp_start), .nlp_seq_len(nlp_seq_len),
        .nlp_busy(nlp_busy), .nlp_done(nlp_done),
        .nlp_class_out(nlp_class_out), .nlp_logit_out(nlp_logit),
        .ext_irq(1'b0),
        .halted(cpu_halted)
    );

    // ---- DistilBERT Accelerator ----
    distilbert_top u_accel (
        .clk(clk), .rst_n(rst_n),
        .nlp_start(nlp_start),
        .nlp_load(nlp_load),
        .nlp_token_data(nlp_token_data),
        .nlp_seq_len(nlp_seq_len),
        .nlp_busy(nlp_busy),
        .nlp_done(nlp_done),
        .nlp_class_out(nlp_class_out),
        .nlp_logit_out(nlp_logit),
        .axi_ar_addr(axi_ar_addr), .axi_ar_len(axi_ar_len),
        .axi_ar_valid(axi_ar_valid), .axi_ar_ready(axi_ar_ready),
        .axi_r_data(axi_r_data), .axi_r_valid(axi_r_valid),
        .axi_r_last(axi_r_last), .axi_r_ready(axi_r_ready),
        .axi_aw_addr(axi_aw_addr), .axi_aw_len(axi_aw_len),
        .axi_aw_valid(axi_aw_valid), .axi_aw_ready(axi_aw_ready),
        .axi_w_data(axi_w_data), .axi_w_valid(axi_w_valid),
        .axi_w_last(axi_w_last), .axi_w_ready(axi_w_ready),
        .axi_b_valid(axi_b_valid), .axi_b_ready(axi_b_ready),
        .current_layer(nlp_layer),
        .current_state()
    );

    assign nlp_class = nlp_class_out;

    // ---- UART TX (stub for result output) ----
    assign uart_tx = 1'b1; // Idle high

endmodule

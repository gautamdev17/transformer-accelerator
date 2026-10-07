`timescale 1ns/1ps
// Checks rtl softmax_int against vectors from softmax_int_ref.py (bit-exact)
module tb_softmax_int;
    reg clk = 0, rst_n = 0, start = 0;
    reg [6:0] seq_len;
    wire [6:0] score_rd_idx, attn_wr_idx, current_row;
    wire signed [31:0] score_in;
    wire score_rd_en, attn_wr_en, done, busy;
    wire [7:0] attn_out;

    reg signed [31:0] score_mem [0:16383];
    reg [7:0]         attn_mem  [0:16383];
    assign score_in = score_mem[{current_row, score_rd_idx}];   // same as attention_head
    always @(posedge clk) if (attn_wr_en) attn_mem[{current_row, attn_wr_idx}] <= attn_out;

    softmax_int dut(.clk(clk),.rst_n(rst_n),.start(start),.seq_len(seq_len),
        .score_rd_idx(score_rd_idx),.score_in(score_in),.score_rd_en(score_rd_en),
        .attn_wr_idx(attn_wr_idx),.attn_out(attn_out),.attn_wr_en(attn_wr_en),
        .current_row(current_row),.done(done),.busy(busy));
    always #5 clk = ~clk;

    reg [31:0] in_v  [0:16383];
    reg [7:0]  exp_v [0:16383];
    integer cfg_fd, n, r, c, in_ptr, exp_ptr, errs, total, tests, rc;
    integer cyc;

    initial begin
        $readmemh("sim/vec_in.hex", in_v);
        $readmemh("sim/vec_exp.hex", exp_v);
        cfg_fd = $fopen("sim/vec_cfg.txt","r");
        in_ptr = 0; exp_ptr = 0; errs = 0; total = 0; tests = 0;
        #22 rst_n = 1;
        while (!$feof(cfg_fd)) begin
            rc = $fscanf(cfg_fd, "%d\n", n);
            if (rc == 1) begin
                seq_len = n;
                for (r = 0; r < n; r = r + 1)
                    for (c = 0; c < n; c = c + 1) begin
                        score_mem[{r[6:0], c[6:0]}] = in_v[in_ptr]; in_ptr = in_ptr + 1;
                    end
                @(posedge clk); start <= 1; @(posedge clk); start <= 0;
                cyc = 0;
                while (!done && cyc < 2000000) begin @(posedge clk); cyc = cyc + 1; end
                @(posedge clk);
                for (r = 0; r < n; r = r + 1)
                    for (c = 0; c < n; c = c + 1) begin
                        total = total + 1;
                        if (attn_mem[{r[6:0], c[6:0]}] !== exp_v[exp_ptr]) begin
                            errs = errs + 1;
                            if (errs < 10) $display("MISMATCH n=%0d row %0d col %0d got %0d exp %0d", n, r, c, attn_mem[{r[6:0],c[6:0]}], exp_v[exp_ptr]);
                        end
                        exp_ptr = exp_ptr + 1;
                    end
                tests = tests + 1;
                $display("config %0d: seq_len=%0d  cycles=%0d  (%0d cyc/row)", tests, n, cyc, cyc/n);
            end
        end
        $display("%s  %0d/%0d outputs match", errs==0 ? "PASS" : "FAIL", total-errs, total);
        $finish;
    end
endmodule

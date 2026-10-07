# Integer-only softmax change (I-BERT style)

Removed:  rtl/attention/softmax_fp16.v, rtl/compute/fp16_adder.v, rtl/compute/fp16_multiplier.v,
          int8_to_fp16 / fp16_to_int8 / fp16_to_int4 (fixed_point_utils.v), FP16_* defines.
Added:    rtl/attention/softmax_int.v, prob_u8_to_int4 (fixed_point_utils.v),
          sim/softmax_int_ref.py (bit-exact model), sim/tb_softmax_int.v
Edited:   attention_head.v (instantiates softmax_int; attn buffer 16b -> 8b),
          av_matmul_int4.v (attn_data 16b -> 8b, uses prob_u8_to_int4), defines.v (comments, FP16 defines)

Run:  python3 sim/softmax_int_ref.py gen
      iverilog -g2005 -o sim/tb.vvp -I rtl/attention -I rtl rtl/attention/softmax_int.v sim/tb_softmax_int.v && vvp sim/tb.vvp

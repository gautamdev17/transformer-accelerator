#!/usr/bin/env python3
"""Bit-exact model of rtl/attention/softmax_int.v  (I-BERT integer softmax).

Also reports error vs. float softmax so you can see what the approximation costs.
Usage:  python3 softmax_int_ref.py gen   -> writes vectors for tb_softmax_int.v
        python3 softmax_int_ref.py err   -> accuracy vs float softmax
"""
import sys, math, random
import numpy as np

Q_LN2, Q_B, Q_C = 177, 346, 62885          # S_in = 2^-8, I-BERT a,b,c
RECIP = 5925

def softmax_int_row(scores):
    m = max(scores)
    exps = []
    for s in scores:
        x = (s - m) >> 3                    # arithmetic shift, <= 0
        u = min(-x, 4095)
        z = (u * RECIP) >> 20               # == u // 177
        r = u - z * Q_LN2                   # -p in [0,176]
        t = Q_B - r
        L = t * t + Q_C
        exps.append(0 if z >= 18 else L >> z)
    tot = sum(exps)
    factor = ((1 << 32) // tot) & 0xFFFF
    return [min(255, (e * factor + (1 << 23)) >> 24) for e in exps]

def softmax_float_row(scores):
    x = (np.array(scores, dtype=np.float64) - max(scores)) / 8.0 / 256.0
    e = np.exp(x); return e / e.sum()

def make_scores(rng, n, kind):
    # raw INT32 QK^T accumulators; real logit = score/8/256
    spread = {"tight": 2000, "mid": 8000, "wide": 30000, "huge": 2_000_000}[kind]
    return [int(v) for v in rng.integers(-spread, spread, n)]

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    if sys.argv[1] == "gen":
        cases = [(8,"mid"),(16,"tight"),(32,"wide"),(8,"huge"),(24,"mid"),(120,"mid")]
        with open("sim/vec_cfg.txt","w") as f, open("sim/vec_in.hex","w") as fi, open("sim/vec_exp.hex","w") as fo:
            for n,k in cases:
                f.write(f"{n}\n")
                for _ in range(n):
                    sc = make_scores(rng, n, k)
                    # also plant duplicates of the max / extreme negative diff
                    if rng.random() < .2: sc[0] = sc[-1] = max(sc)
                    for s in sc: fi.write(f"{s & 0xFFFFFFFF:08x}\n")
                    for a in softmax_int_row(sc): fo.write(f"{a:02x}\n")
        print("vectors written")
    else:
        worst = 0; tot = 0; cnt = 0; rs = 0
        for n,k in [(8,"mid"),(16,"tight"),(32,"wide"),(120,"mid")]:
            for _ in range(300):
                sc = make_scores(rng, n, k)
                a = np.array(softmax_int_row(sc))/256.0
                f = softmax_float_row(sc)
                d = np.abs(a-f); worst = max(worst, d.max()); tot += d.sum(); cnt += n; rs = max(rs, abs(a.sum()-1))
        print(f"max |A_int - A_float| = {worst:.4f}  (1 LSB = {1/256:.4f})")
        print(f"mean abs err         = {tot/cnt:.5f}")
        print(f"max |row_sum - 1|    = {rs:.4f}")

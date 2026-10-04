"""Revision: single-user crafting latency of the amortized generator vs the per-user
iterative attack, measured under identical conditions (one CPU core, idle machine).
Latency does not depend on trained weights, so freshly initialized networks of the
paper's sizes are timed. Run with BEAM_DEVICE=cpu on an otherwise idle machine."""
import os, json, time, numpy as np, torch
import beamdata as bd
import ris
import revision_common as rc
from stage_amortized import PhaseGenerator, gen_input
from model import BeamMLP, DEVICE

REPS_GEN, REPS_IT = 2000, 10


def main():
    torch.set_num_threads(1)
    H = np.load(os.path.join(bd.ART, "channels_64ant.npy"))[:1]
    out = {"threads": 1, "device": str(DEVICE), "M": {}}
    victims = {"rsrp": rc.Victim("rsrp", rc.RSRPMLP(rc.L_WIDE, bd.N_BEAMS).to(DEVICE).eval(), rc.wide_codebook()),
               "csi": rc.Victim("csi", BeamMLP(2 * bd.N_BS_ANT, bd.N_BEAMS).to(DEVICE).eval())}
    for M in (64, 128):
        g = ris.build_geometry(H, M, 1.0)
        G = PhaseGenerator(bd.N_BS_ANT, M).to(DEVICE).eval()
        x = gen_input(g["H_d"], g["k_u"])
        with torch.no_grad():
            for _ in range(50):
                G(x)
            ts = []
            for _ in range(REPS_GEN):
                t0 = time.perf_counter(); G(x); ts.append(time.perf_counter() - t0)
        r = {"gen_median_ms": float(np.median(ts) * 1e3), "gen_p99_ms": float(np.percentile(ts, 99) * 1e3)}
        for kind, v in victims.items():
            ts = []
            for k in range(REPS_IT):
                t0 = time.perf_counter(); rc.ris_attack(v, g, iters=80, seed=k); ts.append(time.perf_counter() - t0)
            r[f"iter_{kind}_median_s"] = float(np.median(ts))
        out["M"][str(M)] = r
        print(M, r, flush=True)
    json.dump(out, open(os.path.join(bd.ART, "stage_latency_gen.json"), "w"), indent=2)
    print("Saved stage_latency_gen.json")


if __name__ == "__main__":
    main()

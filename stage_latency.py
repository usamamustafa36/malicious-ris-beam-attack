"""Review-response: wall-clock cost of crafting ONE user's malicious-RIS phase
config (single-user 80-step Adam optimisation, matching Sec.3.2's default),
compared against Doppler coherence time at 3.5GHz and 28GHz for pedestrian and
vehicular mobility. This CPU-only research-code implementation (no GPU, no
SDR/embedded optimisation) gives an upper bound on attacker compute latency;
CSI-acquisition and RIS phase-actuation time are additional, unmodelled
overheads on top of this number."""
import time, json, os, numpy as np, torch
import beamdata as bd
from model import BeamMLP, DEVICE
import ris

REPEATS = 20


def time_single_user_attack(model, H1, M, iters=80, lr=0.1, reps=REPEATS):
    times = []
    for r in range(reps):
        g = ris.build_geometry(H1, M, 1.0, seed=r)
        t0 = time.perf_counter()
        ris.ris_attack(model, g, iters=iters, lr=lr, seed=r)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.std(times))


def main():
    d = bd.build_dataset()
    H1 = d["H_test"][:1]                 # single user
    m = BeamMLP(d["Xte"].shape[1], d["n_beams"]).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(bd.ART, "victim_model.pt"), map_location=DEVICE)); m.eval()

    out = {"note": "single-user (batch=1) 80-step Adam wall-clock, CPU, median of "
                    f"{REPEATS} repeats; research code, not embedded/SDR-optimised.",
           "M_sweep": {}}
    for M in (64, 128):
        med, std = time_single_user_attack(m, H1, M)
        out["M_sweep"][str(M)] = {"median_s": med, "std_s": std}
        print(f"M={M:3d}: median={med*1000:.1f} ms  std={std*1000:.1f} ms")

    # coherence-time context (physics only, T_c ~ lambda / (2 v), a standard
    # "time to move half a wavelength" order-of-magnitude rule of thumb)
    c = 3e8
    out["coherence_ms"] = {}
    for fc_ghz, band in [(3.5, "3.5GHz"), (28.0, "28GHz")]:
        lam = c / (fc_ghz * 1e9)
        for v, mob in [(1.5, "pedestrian_1.5ms"), (30.0, "vehicular_30ms")]:
            tc = lam / (2 * v)
            out["coherence_ms"][f"{band}_{mob}"] = tc * 1000
            print(f"  T_c[{band}, {mob}] = {tc*1000:.2f} ms")

    json.dump(out, open(os.path.join(bd.ART, "stage_latency.json"), "w"), indent=2)
    print("\nSaved stage_latency.json")


if __name__ == "__main__":
    main()

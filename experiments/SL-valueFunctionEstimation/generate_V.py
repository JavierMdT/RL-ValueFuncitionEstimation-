import math
from pathlib import Path
import numpy as np

ROOT = Path("./V_truth")
GAMMA = 0.99

# ==========================================
# 1. MOUNTAIN CAR REFERENCE GENERATION
# ==========================================
def build_mountaincar_reference():
    NREF = 401
    MAP_N = 201
    PLO, PHI = -1.2, 0.6
    VLO, VHI = -0.07, 0.07
    GOAL = 0.5
    MAX_REF_ITER = 5000

    pgrid = np.linspace(PLO, PHI, NREF)
    vgrid = np.linspace(VLO, VHI, NREF)
    P, V = np.meshgrid(pgrid, vgrid, indexing="ij")
    terminal = P >= GOAL
    
    idx = np.zeros((NREF, NREF, 3, 4), dtype=np.int32)
    weights = np.zeros((NREF, NREF, 3, 4), dtype=np.float64)
    done = np.zeros((NREF, NREF, 3), dtype=bool)
    
    def brackets(grid, x):
        hi = np.searchsorted(grid, x, side="right")
        hi = np.clip(hi, 1, len(grid) - 1)
        lo = hi - 1
        w = (x - grid[lo]) / (grid[hi] - grid[lo])
        lo = np.where(x <= grid[0], 0, lo)
        hi = np.where(x <= grid[0], 0, hi)
        w = np.where(x <= grid[0], 0.0, w)
        lo = np.where(x >= grid[-1], len(grid) - 1, lo)
        hi = np.where(x >= grid[-1], len(grid) - 1, hi)
        w = np.where(x >= grid[-1], 0.0, w)
        return lo, hi, w

    for i, p in enumerate(pgrid):
        for j, v in enumerate(vgrid):
            if terminal[i, j]: continue
            for a in range(3):
                v2 = float(np.clip(v + (a - 1) * 0.001 - 0.0025 * math.cos(3 * p), VLO, VHI))
                p2 = float(np.clip(p + v2, PLO, PHI))
                if p2 == PLO and v2 < 0: v2 = 0.0
                if p2 >= GOAL:
                    done[i, j, a] = True
                    continue
                i0, i1, wp = brackets(pgrid, np.array([p2]))
                j0, j1, wv = brackets(vgrid, np.array([v2]))
                i0, i1, wp = int(i0[0]), int(i1[0]), float(wp[0])
                j0, j1, wv = int(j0[0]), int(j1[0]), float(wv[0])
                inds = [(i0, j0), (i1, j0), (i0, j1), (i1, j1)]
                ws = [(1 - wp) * (1 - wv), wp * (1 - wv), (1 - wp) * wv, wp * wv]
                for k, ((ii, jj), w) in enumerate(zip(inds, ws)):
                    idx[i, j, a, k] = np.ravel_multi_index((ii, jj), (NREF, NREF))
                    weights[i, j, a, k] = w
                    
    val = np.zeros((NREF, NREF), dtype=np.float64)
    for it in range(MAX_REF_ITER):
        cont = np.sum(val.ravel()[idx] * weights, axis=3)
        q = -1.0 + GAMMA * cont
        q[done] = -1.0
        new = np.max(q, axis=2)
        new[terminal] = 0.0
        residual = float(np.max(np.abs(new - val)))
        val = new
        if residual < 1e-9: break
        
    mp = np.linspace(PLO, PHI, MAP_N)
    mv = np.linspace(VLO, VHI, MAP_N)
    MP, MV = np.meshgrid(mp, mv, indexing="ij")
    map_pts = np.column_stack((MP.ravel(), MV.ravel()))
    
    # Interpolation helper for map
    def interp_grid_vals(values, points):
        i0, i1, wp = brackets(pgrid, points[:, 0])
        j0, j1, wv = brackets(vgrid, points[:, 1])
        return ((1 - wp) * (1 - wv) * values[i0, j0] + wp * (1 - wv) * values[i1, j0]
                + (1 - wp) * wv * values[i0, j1] + wp * wv * values[i1, j1])
                
    vmap_ref = interp_grid_vals(val, map_pts).reshape(MAP_N, MAP_N)
    
    np.savez_compressed(
        ROOT / "mountaincar_truth_reference.npz",
        pgrid=pgrid, vgrid=vgrid, vref=val,
        map_points=map_pts, vmap_ref=vmap_ref
    )
    print("-> MountainCar ground truth guardado.")

# ==========================================
# 2. PENDULUM REFERENCE GENERATION (θ, θ_dot)
# ==========================================
def build_pendulum_reference():
    NREF = 151
    th_grid = np.linspace(-np.pi, np.pi, NREF)
    v_grid = np.linspace(-8.0, 8.0, NREF)
    TH, V = np.meshgrid(th_grid, v_grid, indexing="ij")
    
    actions = np.array([-2.0, 0.0, 2.0])
    val = np.zeros((NREF, NREF), dtype=np.float64)
    dt = 0.05
    g = 10.0
    m = 1.0
    l = 1.0
    
    def wrap_angle(theta):
        return ((theta + np.pi) % (2 * np.pi)) - np.pi

    # Value iteration para Pendulum en rejilla 2D (theta, theta_dot)
    for it in range(300):
        new_val = np.copy(val)
        for i, th in enumerate(th_grid):
            for j, v in enumerate(v_grid):
                q_vals = []
                for u in actions:
                    # Dinámica de Pendulum-v1
                    cost = (wrap_angle(th) ** 2) + 0.1 * (v ** 2) + 0.001 * (u ** 2)
                    next_v = np.clip(v + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3 / (m * l**2) * u) * dt, -8.0, 8.0)
                    next_th = wrap_angle(th + next_v * dt)
                    
                    # Interpolar valor del estado siguiente en la rejilla
                    th_idx = np.clip(np.searchsorted(th_grid, next_th) - 1, 0, NREF - 2)
                    v_idx = np.clip(np.searchsorted(v_grid, next_v) - 1, 0, NREF - 2)
                    
                    v_next = val[th_idx, v_idx] # Aproximación simple por vecino/interp
                    q_vals.__(cost + GAMMA * v_next) if hasattr(q_vals, '__') else q_vals.append(-cost + GAMMA * v_next) # Nota: recompensa negativa
                new_val[i, j] = max(q_vals)
        diff = np.max(np.abs(new_val - val))
        val = new_val
        if diff < 1e-4: break

    np.savez_compressed(
        ROOT / "pendulum_truth_reference.npz",
        th_grid=th_grid, v_grid=v_grid, vref=val
    )
    print("-> Pendulum ground truth guardado.")

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    print("Creating mountain car V*...")
    build_mountaincar_reference()
    print("Creating pendulum V*...")
    build_pendulum_reference()
    print("¡Done generating V* ./V_truth/!")

if __name__ == "__main__":
    main()
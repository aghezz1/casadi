import casadi as ca
import numpy as np

rng = np.random.default_rng(0)
n = 100
m = 80
ms = 55
x = ca.SX.sym("x", n)
# --- Hessian: dense symmetric matrix similar to full(sprandsym(...)) ---
M = rng.standard_normal((n, n))
H = 0.5 * (M @ M.T)                         # symmetrize

# --- Linear term ---
f = 100 * rng.standard_normal((n, 1))
# f[:ms] = -np.sign(f[:ms]) * f[:ms]

# --- Constraint matrix and bounds ---
A = rng.standard_normal((m, n))
bupper = 20 * rng.random((m, 1))
blower = -20 * rng.random((m, 1))

solver = ca.qpsol('solver', 'daqp',
                  {'f': 0.5*x.T@H@x + f.T @ x, 'x': x, "g": A@x},
                  {'discrete': [1] * ms + [0] * (n-ms)}
                  )
daqp_sol = solver(lbx=[0] * ms + [-10] * (n-ms), ubx=[1] * ms + [10] * (n-ms),
             lbg=blower, ubg=bupper)
print(f"Optimal solution: {daqp_sol['x'].full().squeeze()}")
print(f"Optimal objective: {float(daqp_sol['f'])}")

solver = ca.qpsol('solver', 'gurobi',
                  {'f': 0.5*x.T@H@x + f.T @ x, 'x': x, "g": A@x},
                  {'discrete': [1] * ms + [0] * (n-ms)}
                  )
gu_sol = solver(lbx=[0] * ms + [-10] * (n-ms), ubx=[1] * ms + [10] * (n-ms),
              lbg=blower, ubg=bupper)
print(f"Optimal solution: {gu_sol['x'].full().squeeze()}")
print(f"Optimal objective: {float(gu_sol['f'])}")

np.allclose(daqp_sol["x"].full().squeeze(), gu_sol["x"].full().squeeze(), rtol=1e-6, atol=1e-6)
np.allclose(float(gu_sol['f']), float(daqp_sol['f']), rtol=1e-6, atol=1e-6)
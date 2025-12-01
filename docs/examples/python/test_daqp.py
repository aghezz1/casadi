import casadi as ca

# Binary decision variables
x = ca.SX.sym("x")
y = ca.SX.sym("y")
z = ca.SX.sym("z")

# Constraints
g_list = []
g_list.append(x + 2*y + 3*z)
lbg = [-ca.inf]
ubg = [4]
g_list.append(x + y)
lbg.append(1)
ubg.append(ca.inf)

# Objective: minimize -(x + y + 2 z)
f = x**2 + y**2 + z**2

solver = ca.nlpsol('solver', 'bonmin',
                  {'f': f, 'g': ca.vertcat(*g_list), 'x': ca.vertcat(x, y, z)},
                  {'discrete': [1, 1, 1]}
                  )
sol = solver(lbx=0, ubx=1, lbg=lbg, ubg=ubg)

solver = ca.qpsol('solver', 'daqp',
                  {'f': f, 'g': ca.vertcat(*g_list), 'x': ca.vertcat(x, y, z)},
                  {'discrete': [1, 1, 1]}
                  )
sol = solver(lbx=0, ubx=1, lbg=lbg, ubg=ubg)
print(f"Optimal solution: {sol['x'].full().squeeze()}")

# A = ca.DM([[1, 2, 3], [1, 1, 0]])
# H = ca.DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


# solver = ca.conic('solver', 'daqp',
#                   {"h": H.sparsity(),
#                     'a': A.sparsity()},
#                   # {'discrete': [1, 1, 1]}
#                   )
# # Solve
# sol = solver(h=H, a=A, lbx=0, ubx=1, lba=lbg, uba=ubg)
# print(f"Optimal solution: {sol['x'].full().squeeze()}")


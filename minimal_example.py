import numpy as np
import matplotlib.pyplot as pl
import wedgeqs

t = np.linspace(0, -10e6, 101)
x0, L = 100e3, 200e3
alpha, beta = 0.05, -0.15
vc, vd, vxa, vya = -8e-3, -2e-3, 0.0, 0.0

x, y = wedgeqs.analytical_solution(t, x0, alpha, beta, L, vc, vd, vxa, vya)

fig, ax = pl.subplots(1, 1)
sc = ax.scatter(x/1000.0, y/1000.0, c=t/1e6, s=5)
cb = fig.colorbar(sc, shrink=0.5)
cb.set_label('Age (Ma)')
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Elevation (km)')
fig.show()
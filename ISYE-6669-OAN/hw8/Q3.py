import numpy as np
import matplotlib.pyplot as plt
from simplex import splex

# Input values for plotting constraint functions
x = np.linspace(0, 4, 100)

# Constraint function values
y1 = 4 - x
y2 = 2*x + 2
y3 = 6-2*x

# Setup grid to plot in and plot constraints
plt.ion()
fig, ax = plt.subplots(1, 1)
ax.set_title('Visualization of Simplex Method Iterations')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.xlim([-1, 4])
plt.ylim([-1, 8])
plt.plot(x, y1, '-b', alpha=0.8)
plt.plot(x[0:int(0.75*len(x))], y2[0:int(0.75*len(y2))], '-r', alpha=0.8)
plt.plot(x[0:int(0.75*len(x))], y3[0:int(0.75*len(y3))], '-g', alpha=0.8)
plt.vlines(0, 0, 8, color='black', linestyles='solid')
plt.hlines(0, 0, 4, color='black', linestyles='solid')

# Shading
y_shading = [min(min(item[0], item[1]), item[2]) for item in zip(y1, y2, y3)]
ybott = np.zeros(len(y_shading))
ax.fill_between(x, y_shading, where=y_shading > ybott, color='orange', alpha=0.5)
plt.annotate('Feasible Region', (0.5, 1.25))

# Defining the optimization problem
A = np.array([[1., 1., 1., 0., 0.], [-2., 1., 0., 1., 0.], [2., 1., 0., 0., 1.]])
b = np.array([[4.], [2.], [6.]])
c = np.array([-1., -2., 0., 0., 0.])

# Simplex initialization parameters
start_basis = [3., 4., 5.]

splex(start_basis, A, b, c)
plt.ioff()
# Holds the final image on screen
plt.show()

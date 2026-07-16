import numpy as np
import matplotlib.pylab as plt

x0 = 1.5
y0 = 1.5

x = np.linspace(0.0, 3.0, 200)
y = np.linspace(0.0, 3.0, 200)
X, Y = np.meshgrid(x, y)

r = 1.0

theta = np.atan2(Y - y0, X - x0)

c1 = 0.25
c2 = 0.5

# lsf = X**2 + Y**2 - r**2 + c * np.cos(2 * theta)
lsf = (
    (X - x0) ** 2
    + (Y - y0) ** 2
    - r**2
    + c1 * np.cos(3 * theta)
    + c2 * np.cos(4 * theta)
)


plt.contour(X, Y, lsf, levels=[0])
plt.axis("equal")
plt.axis("off")

plt.show()

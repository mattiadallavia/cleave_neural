import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 100).reshape((1, 100))
y = np.arange(1000, 0, -10).reshape((1, 100))
plt.plot(x, np.log10(y), 'bo-', linewidth=2)
plt.show()

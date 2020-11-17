import matplotlib.pyplot as plt

x, y, u = [], [], []
for line in open('controller_pid.dat','r'):
    values = [float(s) for s in line.split()]
    x.append(values[0])
    y.append(values[1])
    u.append(values[3])

fig, axs = plt.subplots(2)
fig.suptitle('Angle and Force')
axs[0].plot(x,y)
axs[1].plot(x,u)
plt.show()
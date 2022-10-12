from calendar import different_locale
from cmath import sin
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)

import statsmodels.api as sm


c = 3e2


class Simulator:
    def __init__(self, times):
        self.okay = True
        self.time_interval = 0
        self.min = 0
        self.max = 0
        self.times = times
    
    def wave(self, stat):
        stat[0] += np.abs(sin(stat[0]))
        return stat
    
    def sawtooth(self, T : int, min : int, max : int):
        self.time_interval = T
        self.max = max
        self.min = min

        m = (max-min)/T
        f = (self.times*m)%(max-min) + min
        return f


def FindDifference(t,f, stats ):

    time_deltas = np.zeros(len(t))

    for i, stat in enumerate(stats):
        mfreq = stat[1]

        diff = 100000
        index = i

        while (diff < 0.01 and index > 0):
            if (mfreq == f[index]):
                time_deltas[i] = stat[0] - t[index]
                found = True
            index -= 1

    distance = c*time_deltas
    return distance


start = 40
end  = 120
dt = 0.001
num = (end - start)/dt
t = np.linspace(start, end, int(num))

sim = Simulator(t)
f = sim.sawtooth(5, 2000, 10000)


stats = np.array([t,f])
stats = np.moveaxis(stats, 0, 1)

for stat in stats:
    stat = sim.wave(stat)


distance = FindDifference(t,f,stats)

stats = np.moveaxis(stats, 1,0)

correlation = sm.tsa.stattools.ccf(stats[0], f, adjusted=False)

plt.subplot(2, 2, 1)
plt.plot(stats[0], stats[1])
plt.plot(t,f)

plt.subplot(2,2, 2)
plt.plot(t, distance)

plt.subplot(2,2, 3)
plt.plot(t, correlation)

plt.show()
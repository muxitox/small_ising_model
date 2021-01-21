import numpy as np
from itertools import product
import math
import random
import copy

class SimpleIsingModel:
    def __init__(self, N, lr=0.0005):
        self.lr = lr
        
        # Possible values the spins can take
        self.spin_values = [-1,1]
        
        self.num_spins = N
        
                
        # Calculate the initial H vector, each entry proportional to its probability (according to Hinton)
        # Try with random H
        self.H = np.random.normal(loc=0.0, scale=.01, size=self.num_spins)

    def metropolis_step(self):
            spin_to_flip = np.random.randint(self.num_spins)
            delta_energy = 2*(self.H[spin_to_flip]*self.sigma[spin_to_flip])
            p = math.exp(-delta_energy)
            r = random.random()
            if delta_energy <= 0 or r<p:
                self.sigma[spin_to_flip] = -self.sigma[spin_to_flip]
class ising:
    def __init__(self, netsize):  # Create ising model

        self.size = netsize
        self.H = np.zeros(netsize)
        self.J = np.zeros((netsize, netsize))
        self.randomize_state()
        self.Beta = 1
    def MetropolisStep(self, i=None):  # Execute step of Metropolis algorithm
            if i is None:
                i = np.random.randint(self.size)
            eDiff = self.deltaE(i)
            if eDiff <= 0 or np.random.rand(
                1) < np.exp(-self.Beta * eDiff):    # Metropolis
                self.s[i] = -self.s[i]

    def randomize_state(self):
        self.s = np.random.randint(0, 2, self.size) * 2 - 1
        
    def deltaE(self, i):  # Compute energy difference between two states with a flip of spin i
        return 2 * (self.s[i] * self.H[i] + np.sum(self.s[i] * \
                    (self.J[i, :] * self.s) + self.s[i] * (self.J[:, i] * self.s)))
N=1
I1 = SimpleIsingModel(N)
H= np.random.randn(N)
I1.H=H.copy()
I1.sigma =np.random.randint(0, 2, N) * 2 - 1

I2=ising(N)
I2.H=H.copy()

T=10000
m1=np.zeros(N)
m2=np.zeros(N)
for t in range(T):
    I1.metropolis_step()
    I2.MetropolisStep()
    m1+=I1.sigma/T
    m2+=I2.s/T
m=np.tanh(H)

print(m)
print(m1)
print(m2)

print()
print('error')
print(m-m1)
print(m-m2)

p=(1+m)/2
q=1-p

SE = np.sqrt(p*q/T)
print(SE*3)               # 3 times the standard deviation covers for ~95% of the error






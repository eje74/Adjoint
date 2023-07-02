import numpy as np
import matplotlib.pyplot as plt


def circ_shift(valin, c_input):
    valout = valin
    for d, c in enumerate(c_input):
        valout = np.roll(valout, c, axis=d)
    return valout

def propagate(f, c_input):
    for q, c in enumerate(c_input):
        f[q, ...] = circ_shift(f[q, ...], c)
    return f

def sub2ind(alpha, sub, system_size):
    return 9*int(sub[0]*system_size[1] + np.mod(sub[1], system_size[1])) + alpha

c = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=int)
alpha_hat = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7], dtype=int)
w = np.array([4/9] + [1/9]*4 + [1/36]*4)

system_size = (20, 20)
num_iter = 10000

tau = 0.7

rho_left_hand = 1
rho_right_hand = 0.0

D = (tau - 0.5)/3
flux_max = -D*(rho_right_hand- rho_left_hand)/(system_size[0]-1)

# Init system
phi= 0.5*np.ones(system_size)
g = w.reshape(9,1,1)*phi

# Main loop
for iter in np.arange(num_iter):
    phi = np.sum(g, axis=0)
    g_out = (1-1/tau)*g + (1/tau)*w.reshape(9,1,1)*phi

    # Propagation
    g = propagate(g_out, c)

    # Boundary conditions
    # -- left hand side
    for dir in [5, 1, 7]:
        g[dir, 0, :] = 2*w[dir]*rho_left_hand - g[alpha_hat[dir], 0, :]

    # -- Right hand side
    for dir in [8, 2, 6]:
        g[dir, -1, :] = 2*w[dir]*rho_right_hand - g[alpha_hat[dir], -1, :]

J = np.sum(g[:, 0, :]*c[:,0].reshape(9,1), axis=0)
jx_mean = (1-0.5/tau)*np.sum(J)/system_size[1]


def dg_bulk(alpha, pos, tau, w, c, syst)



print(flux_max, jx_mean)


# Plots
plt.figure()
plt.title(r"$\phi$")
plt.pcolormesh(phi.transpose())
plt.axis("equal")
plt.colorbar()
plt.show()
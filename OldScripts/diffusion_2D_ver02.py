import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from time import time_ns


#########################################################################
#                          Simulations                                  #
#########################################################################

def circ_shift(valin, c_input):
    valout = valin
    for d, c in enumerate(c_input):
        valout = np.roll(valout, c, axis=d)
    return valout

def propagate(f, c_input):
    for q, c in enumerate(c_input):
        f[..., q] = circ_shift(f[..., q], c)
    return f

def source_term(R, tau, xi, w, c, system_size):
    # set up solid geometry
    d = make_geometry(R, system_size)  # Signed distance
    N = normals_from_scalar_field(d)  # Normal vectors
    x = 0.5*(1-np.tanh(d/xi)) # Indicator function
    # Setup the source term
    c = np.array(c)
    c_norm = np.sqrt(np.sum(c**2, axis=-1)) + 10*np.finfo(float).eps
    k = np.sum(w*c_norm)/2/len(system_size)
    lambda_s = 1
    D = (tau - 0.5)/3
    beta = lambda_s*D/(tau*k*xi)
    # broadcasting logic 
    # N   : ni x nj (x 1) x nd 
    # c   :         x nq  x nd
    # c*N : ni x nj x nq
    # w   :           nq
    # x   : ni x nj(x  1)
    #|c|  :           nq 
    #  =  : ni x nj x nq
    boundary_source = w*beta*(x[..., np.newaxis] - 1)*np.sum(c*N[..., np.newaxis, :], axis=-1)/c_norm

    return boundary_source, x

def run_system(lhs, rhs, R, xi, tau, w, c, alpha_rev, number_of_iterations, system_size):
    rho0 = (lhs + rhs)/2

    boundary_source, x = source_term(R, tau, xi, w, c, system_size)

    g = np.zeros(system_size + (w.size,))
    # bulk
    g[1:-1, ...] = w*rho0*x[1:-1, ...,np.newaxis] 
    # lhs
    g[0, ...] = w*lhs*x[0, ..., np.newaxis]
    # lhs
    g[-1, ...] = w*rhs*x[-1, ..., np.newaxis]


    # Main loop
    for itr in np.arange(number_of_iterations):
        # Density
        # -- keepdims=True, broadcasting logic
        #    phi    ni x nj (x  1)
        #    w                nq
        #    w*phi  ni x nj x nq
        phi = np.sum(g, axis=-1, keepdims=True)
        # collision
        g = propagate(
             g - 1/tau*(g - w*phi) + boundary_source*phi,
            c)
        # boundary condition
        # -- left hand side
        for alpha in [1, 5, 7]:
            g[0, :, alpha] = 2*w[alpha]*lhs - g[0, :, alpha_rev[alpha]]
        # -- right hand side
        for alpha in [2, 6, 8]:
            g[-1, :, alpha] = 2*w[alpha]*rhs - g[-1, :, alpha_rev[alpha]]

    # Calculate first moment of the distribution (at the left hand side)
    # -- newaxis broadcasting
    #    g    ni x nj x nq (x 1)
    #    c              nq x nd
    Mi = np.sum(g[0, ..., np.newaxis]*c, axis=-2)    

    # Need to reshape phi from (ni,nj,1) to (ni,nj) 
    return phi.reshape(system_size), (tau-0.5)/tau*np.sum(Mi[:,0])/system_size[1] 

def make_geometry(R, system_size):
    X, Y = np.mgrid[:system_size[0], :system_size[1]]
    X = X-(system_size[0])/2
    Y = Y-(system_size[1])/2
    return R - np.sqrt( X**2 + Y**2 )

def normals_from_scalar_field(phi):
    N = np.zeros( system_size + (phi.ndim,))
    for dim in np.arange(phi.ndim):
        # Derivative for bulk values
        ind_bulk = (slice(None),)*dim + (slice(1, -1), ) + (slice(None),)*(phi.ndim - dim - 1)
        ind_plus = (slice(None),)*dim + (slice(2, None), ) + (slice(None),)*(phi.ndim - dim - 1)
        ind_minus = (slice(None),)*dim + (slice(0, -2), ) + (slice(None),)*(phi.ndim - dim - 1)
        N[ind_bulk + (dim,)] = 0.5*(phi[ind_plus] - phi[ind_minus])
        # Derivative for the left hand boundary
        ind_bulk = (slice(None),)*dim + (0, ) + (slice(None),)*(phi.ndim - dim - 1)
        ind_plus = (slice(None),)*dim + (1, ) + (slice(None),)*(phi.ndim - dim - 1)
        N[ind_bulk + (dim,)] = (phi[ind_plus] - phi[ind_bulk])
        # Derivative for the right hand boundary
        ind_bulk = (slice(None),)*dim + (-1, ) + (slice(None),)*(phi.ndim - dim - 1)
        ind_minus = (slice(None),)*dim + (-2, ) + (slice(None),)*(phi.ndim - dim - 1)
        N[ind_bulk + (dim,)] = (phi[ind_bulk] - phi[ind_minus])
    N /= np.sqrt( np.sum(N**2, axis=-1, keepdims=True) ) + 1e-10
    return N


#########################################################################
#                            Adjoint                                    #
#########################################################################
def dg_dphi(pos, w_list): 
    for alpha in np.arange(w_list.size):
        yield 1, (alpha, pos)

def dg_dj(pos, c_list): 
    for alpha, c in enumerate(c_list):
        yield c[0], (alpha, pos)

def dg_source(alpha, pos, source, w_list):
    pos = np.mod(pos, source.shape[:-1])
    tmp = source[tuple(pos) + (alpha,)]
    for v, i in dg_dphi(pos, w_list):
        yield tmp*v, i

def dg_collision(alpha, pos, tau, source, w_list): 
    yield 1-1/tau, (alpha, pos) 
    for v, i in dg_dphi(pos, w_list):
        yield w_list[alpha]*v/tau, i

    for v, i in dg_source(alpha, pos, source, w_list):
        yield v, i

def dg_dl(alpha, pos, tau, source, w_list, c_list): 
    yield 1, (alpha, pos) 
    for v, i in dg_collision(alpha, pos-c_list[alpha], tau, source, w_list):
        yield -v, i

def dg_dl_boundary(alpha, pos, alpha_hat): 
    yield 1, (alpha, pos)
    yield 1, (alpha_hat[alpha], pos)

def dg_dH(pos, tau, flux, flux_target, c_list, system_size): 
    a = 2*(flux-flux_target)*(tau-0.5)/tau/system_size[1]
    for v, i in dg_dj(pos, c_list):
        yield a*v, i


def dR_source(alpha, pos, xi, phi, x, source):
    pos = np.mod(pos, source.shape[:-1])
    tmp = source[tuple(pos) + (alpha,)]
    sub = tuple(pos)
    ind = sub + (alpha,)
    return 2*source[ind]*x[sub]*phi[sub]/xi

def dR_collision(alpha, pos, xi, phi, x, source):
    return dR_source(alpha, pos, xi, phi, x, source)

def dR_dl(alpha, pos, xi, phi, x, source, c_list):
    return -dR_collision(alpha, pos-c_list[alpha], xi, phi, x, source)

def sub2ind(alpha, sub, system_size, w):
    pos = np.mod(sub, system_size)
    return w.size*int(pos[0]*system_size[1] + pos[1]) + alpha


def adjoint_method(R, tau, xi, phi, j, j_target, w_list, c_list, alpha_hat):
    # Setup system size
    system_size = phi.shape

    source, x_indicator = source_term(R, tau, xi, w_list, c_list, system_size)

    N = 9*np.prod(system_size)

    M = lil_matrix((N, N))
    b = np.zeros((N,))
    dRdl = np.zeros((N,)) 

    for x in np.arange(1, 20):
        for y in np.arange(21):
            pos = np.array([x, y], dtype=int)
            for alpha in np.arange(len(w_list)):
                n = sub2ind(alpha, pos, system_size, w_list)
                dRdl[n] = dR_dl(alpha, pos, xi, phi, x_indicator, source, c_list)
                for v, i in dg_dl(alpha, pos, tau, source, w_list, c_list):
                    m = sub2ind(*i, system_size, w_list)
                    M[m, n] += v


    # Add boundary conditions
    # -- left hand side
    for y in np.arange(21):
        pos = np.array([0, y], dtype=int)
        for v, i in dg_dH(pos, tau, j, j_target, c_list, system_size):
            m = sub2ind(*i, system_size, w_list)
            b[m] -= v
        for alpha in np.arange(len(w_list)):
            n = sub2ind(alpha, pos, system_size, w_list)
            if alpha in [1, 5, 7]:
                for v, i in  dg_dl_boundary(alpha, pos, alpha_hat):
                    m = sub2ind(*i, system_size, w_list)
                    M[m, n] += v
            else:
                dRdl[n] = dR_dl(alpha, pos, xi, phi, x_indicator, source, c_list)
                for v, i in dg_dl(alpha, pos, tau, source, w_list, c_list):
                    m = sub2ind(*i, system_size, w_list)
                    M[m, n] += v
    # -- right hand side.
    for y in np.arange(21):
        pos = np.array([20, y], dtype=int)
        for alpha in np.arange(len(w_list)):
            n = sub2ind(alpha, pos, system_size, w_list)
            if alpha in [2, 6, 8]:
                for v, i in dg_dl_boundary(alpha, pos, alpha_hat):
                    m = sub2ind(*i, system_size, w_list)
                    M[m, n] += v
            else:
                dRdl[n] = dR_dl(alpha, pos, xi, phi, x_indicator, source, c_list)
                for v, i in dg_dl(alpha, pos, tau, source, w_list, c_list):
                    m = sub2ind(*i, system_size, w_list)
                    M[m, n] += v

    M = M.tocsr()
    lambda_l = spsolve(M, b) # Lagrange multiplicator

    return dRdl.dot(lambda_l)


if __name__ == "__main__":
    # Lattice
    c_list = [np.array(c, dtype=int) for c in [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, -1), (1, -1), (-1, 1)
        ]]
    w_list = np.array((4/9,) + 4*(1/9,) + 4*(1/36,))
    alpha_reverse = np.array( [0, 2, 1, 4, 3, 6, 5, 8, 7])

    # Input
    tau_input = 0.7
    lhs = 1
    rhs = 0
    number_of_iterations = 2000
    system_size = (21, 21)

    xi = 0.8
    j_max = (tau_input - 0.5)/3/(system_size[0] - 1)
    j_target = 0.0022871
    number_of_containers = 10

    # Containers for plots
    dH_adjoint = np.zeros(number_of_containers)
    dH_sim = np.zeros(number_of_containers)
    R_x = np.zeros(number_of_containers)

    # Parameter list for 'run_simulation'
    rs_param = (xi, tau_input, w_list, c_list, alpha_reverse, number_of_iterations, system_size)

    for n, R in enumerate(np.linspace(1, 7, number_of_containers)):
        # Adjoint method
        start_time = time_ns()
        phi, j = run_system(lhs, rhs, R, *rs_param)
        print("run_system = ", (time_ns() - start_time)/1e9, " s")
        start_time = time_ns()
        dH_dR = adjoint_method(R, tau_input, xi, phi, j, j_target, w_list, c_list, alpha_reverse)
        print("adjoint_method = ", (time_ns() - start_time)/1e9, " s")        
        dH_adjoint[n] = dH_dR

        # Direct numerical estimate of the variation
        H0 = (j-j_target)**2
        dx = 0.01
        start_time = time_ns()
        phi1, j1 = run_system(lhs, rhs, R + dx, *rs_param)
        print("run_system (2nd run) = ", (time_ns() - start_time)/1e9, " s")    
        print("---------------------------------------------")    
        H1 = (j1-j_target)**2

        # Update data container
        R_x[n] = R
        dH_sim[n] = (H1-H0)/dx


    plt.figure()
    plt.plot(R_x, dH_adjoint, '-', label="adjoint")
    plt.plot(R_x, dH_sim, '--', label="simulation")
    plt.legend()
    plt.show()






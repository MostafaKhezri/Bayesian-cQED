import numpy as np
import multiprocessing as mp
import math
import cmath

"""Calculate Bayesian evolution of a qubit measured in a cQED setup.

For detailed theory, see arXiv:1606.07162.
NOTE: This code is written for the phase sensitive measurement and field reflection configuration.

Variables throughout the code are as follows:
omega_r: resonator frequency, omega_d: drive frequency, chi: resonator response to qubit state
kappa: resonator bandwidth, epsilon: drive strensght, eta: measurement efficiency
T1: qubit relaxation time, T2: qubit coherence time, alpha_p: measurment pump
"""

def RK4update(alpha_old, omega_r, omega_d, chi, kappa, epsilon, dt):
    """Update the resonator field in each timestep using Runge-Kutta-4."""
    alpha_0_old, alpha_1_old = alpha_old
    alpha_0_dot = lambda a: -1j*(omega_r - chi - omega_d)*a - kappa*a/2 - 1j*epsilon
    alpha_1_dot = lambda a: -1j*(omega_r + chi - omega_d)*a - kappa*a/2 - 1j*epsilon
    
    k0_1 = alpha_0_dot(alpha_0_old); k0_2 = alpha_0_dot(alpha_0_old + dt*k0_1/2);
    k0_3 = alpha_0_dot(alpha_0_old + dt*k0_2/2); k0_4 = alpha_0_dot(alpha_0_old + dt*k0_3);
    
    k1_1 = alpha_1_dot(alpha_1_old); k1_2 = alpha_1_dot(alpha_1_old + dt*k1_1/2);
    k1_3 = alpha_1_dot(alpha_1_old + dt*k1_2/2); k1_4 = alpha_1_dot(alpha_1_old + dt*k1_3);
    
    alpha_0_new = alpha_0_old + dt*(k0_1 + 2*k0_2 + 2*k0_3 + k0_4)/6
    alpha_1_new = alpha_1_old + dt*(k1_1 + 2*k1_2 + 2*k1_3 + k1_4)/6
    
    return [alpha_0_new, alpha_1_new]

def pick_I(rho_00, D, Delta_I):
    """Get the noisy detector output."""
    choice = np.random.random()
    if choice <= rho_00:
        out = np.random.normal(-Delta_I/2, math.sqrt(D))
    else:
        out = np.random.normal(Delta_I/2, math.sqrt(D))
    return out

def update(rho_old, alpha_old, omega_r, omega_d, chi, kappa, epsilon, alpha_p, eta, T1, T2, dt, jump):
    """Update qubit and resonator states for each timestep."""
    alpha_0_old, alpha_1_old = alpha_old
    T_phi = 1/(1/T2 - 1/2/T1) #qubit intrinsic dephasing rate
    
    #Calculate parameters from the resonator state
    alpha_0_out, alpha_1_out = math.sqrt(kappa)*alpha_0_old + 1j*epsilon/math.sqrt(kappa), math.sqrt(kappa)*alpha_1_old + 1j*epsilon/math.sqrt(kappa) #output field for reflected config
    I_0, I_1 = abs(alpha_p + alpha_0_out*math.sqrt(dt))**2/dt , abs(alpha_p + alpha_1_out*math.sqrt(dt))**2/dt #output currents
    D = abs(alpha_p)**2/(dt**2) #SI/(2*dt)
    phi_d = cmath.phase(alpha_p) - cmath.phase(alpha_1_out - alpha_0_out) #Angle between amplified quad and information carrying quad
    Delta_I = I_1 - I_0; Delta_I_max = Delta_I/math.cos(phi_d); #response and maximum response
    
    #Calculate Bayesian evolution parameters
    Gamma_d = kappa*abs(alpha_1_old - alpha_0_old)**2/2 #qubit ensemble dephasing rate
    gamma = (1 - eta)*Gamma_d #Total dephasing rate
    d_omega_q = (alpha_1_out.conjugate()*alpha_0_out).imag #AC-Stark shift of the qubit
    K = Delta_I_max*math.sin(phi_d)/(2*D*dt) #Phase back-action term
    
    #Get current and probabilities
    rho_00_old, rho_10_old = rho_old
    Im = pick_I(rho_00_old, D, Delta_I) #Pick the detector output
    P0, P1 = 1/math.sqrt(2*np.pi*D)*math.exp(-(Im + Delta_I/2)**2/(2*D)), 1/math.sqrt(2*np.pi*D)*math.exp(-(Im - Delta_I/2)**2/(2*D))    
    
    #Bayesian update
    rho_00_new = (rho_00_old*P0)/(rho_00_old*P0 + (1-rho_00_old)*P1)
    rho_10_new = rho_10_old*math.sqrt(P0*P1)/(rho_00_old*P0 + (1-rho_00_old)*P1) * math.exp(-gamma*dt) * cmath.exp(-1j*d_omega_q*dt) * cmath.exp(-1j*K*Im*dt) * math.exp(-dt/T_phi)
    
    #T1 update via jump no-jump process
    P_jump = (1 - rho_00_old)*(1 - math.exp(-dt/T1)); #probability of jump at each time step
    choice = np.random.random();
    if choice < P_jump: #state update when jump occurs
        jump = 1;
        rho_00_new = 1.0
        rho_10_new = 0.0
        alpha_old = alpha_1_old, alpha_1_old
    else: #state update when no jump occurs
        rho_00_new = rho_00_new/(1 - P_jump)
        rho_10_new = rho_10_new*math.exp(-dt/T1/2)/(1 - P_jump)
    
    # Update the intracavity state of the resonator
    alpha_0_new, alpha_1_new = RK4update(alpha_old, omega_r, omega_d, chi, kappa, epsilon, dt)
    alpha_1_new = alpha_1_new*(1-jump) + alpha_1_old*jump #Stop evolving alpha_1
    
    return [[rho_00_new, rho_10_new],[alpha_0_new, alpha_1_new], jump]

def evol(rho_init, alpha_init, omega_r, omega_d, chi, kappa, epsilon, alpha_p, eta, T1, T2, dt, T):
    """Calculate time evolution of the qubit and resonator."""
    np.random.seed() #Seeding necesary for parallel calculations using pools
    rho = rho_init; alpha = alpha_init; jump = 0;
    tlist = np.arange(int(T/dt))
    result_rho = np.ndarray(shape = (len(tlist), 2), dtype = complex)
    result_alpha = np.ndarray(shape = (len(tlist), 2), dtype = complex)
    result_rho[0] = np.array(rho_init); result_alpha[0] = np.array(alpha_init);
    for i in range(1, len(tlist)):
        rho, alpha, jump = update(rho, alpha, omega_r, omega_d, chi, kappa, epsilon, alpha_p, eta, T1, T2, dt, jump)
        result_rho[i] = np.array(rho); result_alpha[i] = np.array(alpha);
    rho_00_list, rho_10_list = result_rho.T
    alpha_0_list, alpha_1_list = result_alpha.T;
    return [tlist*dt, rho_00_list.real, rho_10_list, alpha_0_list, alpha_1_list]

def parallelevol(rho_init, alpha_init, omega_r, omega_d, chi, kappa, epsilon, alpha_p, eta, T1, T2, dt, T, trajs):
    """Calculate time evolution of the qubit and resonator for 'trajs' number of trajectories."""
    pool = mp.Pool(processes=8) # This number of processes should be enough for a CPU with up to 8 physical cores.
    results = pool.starmap(evol, np.array([(rho_init, alpha_init, omega_r, omega_d, chi, kappa, epsilon, alpha_p, eta, T1, T2, dt, T)]*trajs))
    results = np.array(results)
    pool.close(); pool.join()
    tlist, all_rho_00, all_rho_10, all_alpha_0, all_alpha_1 = results[0,0].real, results[:,1].real, results[:,2], results[:,3], results[:,4]
    return [tlist, all_rho_00, all_rho_10, all_alpha_0, all_alpha_1]

from matplotlib import pyplot as plt
import numpy as np
from math import sqrt, pi

m_1 = np.array([0.500, 0.510, 0.496, 0.502]) / 1e3
m_2 = np.array([0.518, 0.512, 0.509, 0.510]) / 1e3
dx_1 = np.array([11, 11.5, 10.75, 11.5]) / 1e3
dx_2 = np.array([9.5, 12.25, 10, 9.75]) / 1e2
L = 223 / 1e2
M = 2.905
g = 9.815
sigma_M = 5 / 1e3
sigma_L = 0.5 / 1e3
sigma_m = 0.001 / 1e3
sigma_x_1 = 0.25 / 1e3
sigma_x_2 = 0.25 / 1e2
e_m_1 = sigma_m / m_1
e_m_2 = sigma_m / m_2
e_L = sigma_L / L
e_x_1 = sigma_x_1 / dx_1
e_x_2 = sigma_x_2 / dx_2
e_M = sigma_M / M
u_1 = M / m_1 * sqrt(g) / sqrt(L) * dx_1
e_u_1 = (e_x_1 ** 2 + 0.25 * e_L ** 2 + e_m_1 ** 2 + e_M ** 2) ** 0.5
sigma_u_1 = (e_u_1 * u_1)
for i in range(len(m_1)):
    print(f'Скорость пули {i}: {u_1[i]:.3f} +- {sigma_u_1[i]:.3f} м/c')


R = 32.5 / 1e2
L_1 = 59.8 / 1e2
M_1 = (730.3 + 730.5) / 1e3
t_2 = 92.94
t_1 = 121.91
T_1 = t_1 / 10
T_2 = t_2 / 10
sigma_M_1 = 2 * 5 / 1e3
sigma_t = 383 * 2 / 1e3
sigma_T = sigma_t / 10
e_M_1 = sigma_M_1 / M_1
e_T_1 = sigma_T / T_1
e_T_2 = sigma_T / T_2
e_L_1 = sigma_L / L_1
e_R = sigma_L / R
ki_sqrt = 4 * pi / (T_1 - T_2) * M_1 * R / (T_1 + T_2) * R * T_1
e_ki_sqrt = (e_M_1 ** 2 + 4 * e_R ** 2 + e_T_1 ** 2 + 4 * (e_T_1 + e_T_2)**2) ** 0.5
sigma_ki_sqrt = e_ki_sqrt * ki_sqrt
u_2 = dx_2 *  ki_sqrt / 2 / m_2 / L_1 / R
e_u_2 = (e_L_1 ** 2 + e_m_2 ** 2 + e_R ** 2 + e_ki_sqrt ** 2 + e_x_2 ** 2) ** 0.5
sigma_u_2 = u_2 * e_u_2
for i in range(len(m_2)):
    print(f'Скорость пули {i + 4}: {u_2[i]:.3f} +- {sigma_u_2[i]:.3f} м/c')
from matplotlib import pyplot as plt
import numpy as np
from math import pi, sqrt, ceil


#  Постоянные и предварительные вычисления
L = 214.3 / 1e2  # Длина нити, м
R = 114.6 / 1e3  # Расстояние от центра нижней платформы до крепления нити
r = 30.2 / 1e3   # Расстояние от центра верхней платформы до крепления нити
g = 9.815  # Ускорение свободного падения
m_plat = 965.7 / 1e3  # Масса платформы
m_disk = 584.7 / 1e3  # Масса диска
m_bar = 1273.0 / 1e3  # Масса стержня
sigma_m = 0.5 / 1e3   # Погрешность массы
sigma_L = 0.5 / 1e3   # Погрешность дины нити
sigma_r, sigma_R = sigma_L, sigma_L  # Погрешности рассояния от центра
z_0 = sqrt(L ** 2 - (R - r) ** 2)  # Выстота подвеса
e_z_0 = (4 * (sigma_L / L) ** 2 + 4 * ((sigma_R + sigma_r) / (R - r)) ** 2) ** 1/4  # Отн. погрешность высоты подвеса
sigma_z_0 = e_z_0 * z_0  # Абс. погрещность высоты подвеса
print('Высота подвеста z_0 = ' + str(round(z_0, 6)) + ' +- ' + str(round(sigma_z_0, 6)) + ' м ')  # Выстота подвеса
k = g / 4 * r / pi * R / pi / z_0  # Коэффициент
e_k = sqrt((sigma_r / r) ** 2 + ((sigma_R / R) + (sigma_z_0 / z_0))) ** 2  # Отн. погрешность k
sigma_k = e_k * k  # Абс. погрешность k
print('Коэффициент k = ' + str(round(k, 6)) + ' +- ' + str(round(sigma_k, 6)))


#  Предварительные измерения
eps = 5 / 1e3  #  Требуемая точность
n_1 = 5  # Количество колебаний в предварительном измерении
t_pr = np.array([22.599, 22.294, 22.738, 22.260, 22.591])  # Результаты предварительных измерений
sigma_T = np.std(t_pr)  # Погрешность определения времени
T_pr = np.mean(t_pr) / n_1  # Предварительное вычисление периода
n = ceil(sigma_T / eps / T_pr)  #  Необхоодимое количество колебаний
print('Необходимо ' + str(n) + ' колебаний \n')


#  Определение моментов инерции

#  Платформа
t_plat = 39.337
T_plat = t_plat / n
I_plat = k * m_plat * T_plat ** 2 
eps_I_plat = sqrt((sigma_k / k) ** 2 + (sigma_m / m_plat) ** 2 + 4 * (eps) ** 2)
sigma_I_plat = eps_I_plat * I_plat
print('Момент инерции платформы ' + str(round(I_plat, 6)) + ' +- ' + str(round(sigma_I_plat, 6)))

#  Диск
t_p_d = 35.183
T_p_d = t_p_d / n
I_d_p = k * (m_plat + m_disk) * T_p_d ** 2 
eps_I_d_p = sqrt((sigma_k / k) ** 2 + (2 *sigma_m / (m_plat + m_disk)) ** 2 + 4 * (eps) ** 2)
sigma_I_d_p = eps_I_d_p * I_d_p
I_disk = I_d_p - I_plat
sigma_I_disk = sigma_I_d_p + sigma_I_plat
print('Момент инерции платформы ' + str(round(I_disk, 6)) + ' +- ' + str(round(sigma_I_disk, 6)))

#  Стержень
t_p_b = 33.171
T_p_b = t_p_b / n
I_d_b = k * (m_plat + m_bar) * T_p_b ** 2 
eps_I_d_b = sqrt((sigma_k / k) ** 2 + (2 *sigma_m / (m_plat + m_bar)) ** 2 + 4 * (eps) ** 2)
sigma_I_d_b = eps_I_d_b * I_d_b
I_bar = I_d_b - I_plat
sigma_I_bar = sigma_I_d_b + sigma_I_plat
print('Момент инерции платформы ' + str(round(I_bar, 6)) + ' +- ' + str(round(sigma_I_bar, 6)))

#  Стержень + диск
t_p_b_d = 31.956
T_p_b_d = t_p_b_d / n
I_d_b_d = k * (m_plat + m_bar + m_disk) * T_p_b_d ** 2 
eps_I_d_b_d = sqrt((sigma_k / k) ** 2 + (2 *sigma_m / (m_plat + m_bar + m_disk)) ** 2 + 4 * (eps) ** 2)
sigma_I_d_b_d = eps_I_d_b_d * I_d_b_d
I_bar_disk = I_d_b_d - I_plat
sigma_I_bar_disk = sigma_I_d_b_d + sigma_I_plat
print('Момент инерции платформы ' + str(round(I_bar_disk, 6)) + ' +- ' + str(round(sigma_I_bar_disk, 6)))
print()

#  Вывод
print('I_disk + I_bar = ' + str(round(I_disk + I_bar, 6)) + ' +- ' + str(round(sigma_I_bar + sigma_I_disk, 6)))
print('I_bar_disk = ' + str(round(I_bar_disk, 6)) + ' +- ' + str(round(sigma_I_bar_disk, 6)))
if abs(I_disk + I_bar - I_bar_disk) <= (sigma_I_bar + sigma_I_disk + sigma_I_bar_disk):
    print('Результаты совпадают в пределах погрешности, аддитивность момента инерции подтверждена')


#  Эксперимент с разрезанным цилиндром

#  Вычисление моментов инерции
m = (770.7 + 766.0) / 1e3
h = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
t = np.array([27.349, 27.536, 28.220, 29.258, 30.901, 32.176, 34.045])
h = h / 1e3
h_2 = h ** 2
T = t / n
I, I_2 = [], []
for i in range(len(h)):
    I.append(k * (m_plat + m) * T[i] ** 2 - I_plat)
    I_2.append(I[i] * I[i])

print('Моменты инерции разрезанного цилиндра')
print(I)
print()

# Метод наименьших квадратов

k = (np.mean(h_2 * I) - (np.mean(h_2) * np.mean(I))) / (np.mean(h_2 ** 2) - np.mean(h_2) ** 2)
b = np.mean(I) - k * np.mean(h_2)
sigma_k = 1 / sqrt(len(I)) * sqrt((np.mean(I_2) - np.mean(I) ** 2)/ (np.mean(h_2 ** 2) - (np.mean(h_2) ** 2)) - k)
sigma_b = sigma_k * sqrt(np.mean(h_2 ** 2) - (np.mean(h_2) ** 2))
print('Масса цилиндра m = ' + str(round(k, 6)) + ' +- ' + str(round(sigma_k, 6)))
print('Момент инерции цилиндпра m = ' + str(round(b, 6)) + ' +- ' + str(round(sigma_b, 6)))


# Построение графика
plt.figure(figsize=(12, 9), dpi=100)
plt.ylabel('h^2, м^2')
plt.xlabel('I, кг * м^2')
plt.grid(True, linestyle="--")
plt.title('График зависимости момента инерции разрезанного цилиндра от расстояния каждой половины до оси вращения')
plt.scatter(h_2, I)
plt.plot(h_2, k * h_2 + b, "-r", linewidth=1, label="Линейная аппроксимация I(h^2) = " + str((round(k, 6))) + " h^2 + " + str(round(b, 6)))
plt.legend()
plt.show()





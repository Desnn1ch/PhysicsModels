import math
import matplotlib.pyplot as plt
import numpy as np

V_0_y = 76
H_0 = 280

class CrashSimulation:
    def __init__(self, mass_main=2150, gravity=1.62, exhaust_velocity=3660, fuel_mass=150, fuel_ejection_rate=15):
        self.M = mass_main
        self.g = gravity
        self.V_p = exhaust_velocity
        self.m = fuel_mass
        self.V_m = fuel_ejection_rate
        self.V_0_y = V_0_y
        self.H_0 = H_0
        self.V_0 = 0
        self.h_criticalspeed = 0
        self.t = 0
        self.t_0 = 0

    def calculate_t_0(self):
        self.t_0 = (self.V_0 - self.V_0_y) / self.g
        return self.t_0

    def calculate_height(self, t):
        initial_term = self.H_0 + (self.V_0_y ** 2 / (2 * self.g))
        dynamic_term = self.V_m / ((self.m + self.M) * math.log((self.m + self.M - self.V_m * t) / (self.m + self.M)))
        velocity_term = self.V_p + (self.V_m * (self.V_p * t) + (self.g * t**2) / 2) / ((self.m + self.M) * math.log((self.m + self.M - self.V_m * t) / (self.m + self.M)))
        discriminant = 2 * self.g

        h1 = ((2 * dynamic_term * velocity_term - discriminant) + \
              math.sqrt((2 * dynamic_term * velocity_term - discriminant)**2 - 4 * (dynamic_term**2) * ((velocity_term**2) - discriminant * initial_term))) / (2 * (dynamic_term**2))

        h2 = ((2 * dynamic_term * velocity_term - discriminant) - \
              math.sqrt((2 * dynamic_term * velocity_term - discriminant)**2 - 4 * (dynamic_term**2) * ((velocity_term**2) - discriminant * initial_term))) / (2 * (dynamic_term**2))

        return h1 if 0 < h1 < self.H_0 else h2

    def calculate_velocity(self, t):
        height_at_t = self.calculate_height(t)
        return math.sqrt(self.V_0_y**2 + 2 * self.g * (self.H_0 - height_at_t)) + \
               self.g * t + \
               t * self.V_m * (math.sqrt(self.V_0_y**2 + 2 * self.g * (self.H_0 - height_at_t)) - self.V_p) / (self.m + self.M - self.V_m * t)

    def find_initial_conditions(self):
        for t in np.linspace(0.001, 10, 10000):
            h_test = self.calculate_height(t)
            velocity_test = self.calculate_velocity(t)
            if 0 < h_test <= self.H_0 and 0 <= velocity_test <= 3:
                self.V_0 = math.sqrt(self.V_0_y**2 + 2 * self.g * (self.H_0 - h_test))
                self.h_criticalspeed = h_test
                self.t = t
                return f"h = {h_test:.2f}, v = {velocity_test:.2f}, t = {t:.2f}"

    def velocity(self, t):
        if t < self.t_0:
            return self.V_0_y + self.g * t
        return self.V_0 + self.g * (t - self.t_0) + (t - self.t_0) * self.V_m * (self.V_0 - self.V_p) / (self.m + self.M - self.V_m * (t - self.t_0))

    def acceleration(self, t):
        if t < self.t_0:
            return self.g
        return ((self.m + self.M) * self.V_m * (self.V_0 - self.V_p)) / ((self.m + self.M - self.V_m * (t - self.t_0))**2) + self.g

    def height(self, t):
        if t < self.t_0:
            return self.H_0 - (self.V_0_y * t + self.g * t**2 / 2)
        return self.h_criticalspeed - self.calculate_height(t - self.t_0)


simulation = CrashSimulation()
initial_conditions = simulation.find_initial_conditions()
simulation.calculate_t_0()

time_points = np.linspace(0, simulation.t_0 + simulation.t, 10000)
accelerations = [simulation.acceleration(t) for t in time_points]
velocities = [simulation.velocity(t) for t in time_points]
heights = [simulation.height(t) for t in time_points]

print(f"Итоговые условия: {initial_conditions}")
print(f"Финальная скорость: {simulation.velocity(time_points[-1]):.2f} м/с")

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time_points, heights, label="Высота", color="blue")
plt.title("Высота от времени")
plt.xlabel("Время, с")
plt.ylabel("Высота, м")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_points, velocities, label="Скорость", color="orange")
plt.title("Скорость от времени")
plt.xlabel("Время, с")
plt.ylabel("Скорость, м/с")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time_points, accelerations, label="Ускорение", color="green")
plt.title("Ускорение от времени")
plt.xlabel("Время, с")
plt.ylabel("Ускорение, м/с^2")
plt.grid(True)

plt.tight_layout()
plt.show()

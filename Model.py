import matplotlib.pyplot as plt
import numpy as np
import math

e_0 = 8.85 / (10**12)
q = 1.602 / (10**19)
m = 9.109 / (10**31)
r = 1.5 / 100
R = 4 / 100
V = 8.5 * (10**6)
L = 12 / 100
T = L / V
d = R - r
h = d / 2
delta_t = T / 1000

class Model:
    def __init__(self):
        self.time = 0
        self.position = h
        self.velocity_y = 0
        self.precision = 11
        self.charge = 1

    def voltage(self):
        return self.charge * (10**(-self.precision)) * math.log(R / r) / (2 * math.pi * e_0 * L)

    def acceleration(self):
        return - (self.charge * (10**(-self.precision)) * q) / (2 * math.pi * (r + self.position) * e_0 * L * m)

    def update_velocity(self):
        old_velocity = self.velocity_y
        self.velocity_y += self.acceleration() * delta_t
        return old_velocity

    def update_position(self):
        return self.position + self.update_velocity() * delta_t + self.acceleration() * (delta_t**2) / 2

    def final_velocity(self):
        return math.sqrt(V**2 + self.velocity_y**2)

    def reset(self):
        self.time = 0
        self.position = h
        self.velocity_y = 0

    def simulate(self):
        positions = []
        accelerations = []
        velocities = []

        while True:
            while self.time < T:
                self.time += delta_t
                accelerations.append(self.acceleration())
                velocities.append(self.velocity_y)
                self.position = self.update_position()
                positions.append(self.position)

            if self.position < 0:
                self.charge -= 0.0001
                if self.charge < 0:
                    break
                self.reset()
                positions.clear()
                accelerations.clear()
                velocities.clear()
                continue
            else:
                print("Final Results:")
                print(f"V : {self.final_velocity()}")
                print(f"U : {self.voltage()}")
                break

        positions.pop(-1)
        accelerations.pop(-1)
        velocities.pop(-1)

        time_values = np.linspace(0, T, 1000)
        x_values = np.linspace(0, T * V, 1000)

        self.plot_all_graphs(time_values, x_values, positions, accelerations, velocities)

    @staticmethod
    def plot_all_graphs(time_values, x_values, positions, accelerations, velocities):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        axs[0, 0].set_title("y(t)")
        axs[0, 0].set_xlabel("t")
        axs[0, 0].set_ylabel("y")
        axs[0, 0].plot(time_values, positions, color='blue')
        axs[0, 0].grid(True)

        axs[0, 1].set_title("a_y(t)")
        axs[0, 1].set_xlabel("t")
        axs[0, 1].set_ylabel("a")
        axs[0, 1].plot(time_values, accelerations, color='red')
        axs[0, 1].grid(True)

        axs[1, 0].set_title("v_y(t)")
        axs[1, 0].set_xlabel("t")
        axs[1, 0].set_ylabel("v")
        axs[1, 0].plot(time_values, velocities, color='green')
        axs[1, 0].grid(True)

        axs[1, 1].set_title("y(x)")
        axs[1, 1].set_xlabel("x")
        axs[1, 1].set_ylabel("y")
        axs[1, 1].plot(x_values, positions, color='purple')
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

model = Model()
model.simulate()
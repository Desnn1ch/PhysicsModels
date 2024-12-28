import numpy as np
import matplotlib.pyplot as plt

GRAVITY = 1.62
MASS_CRAFT = 2150
FUEL_INITIAL = 150
EXHAUST_VELOCITY = 3660
FUEL_FLOW_RATE = 15
INITIAL_VELOCITY = 76
INITIAL_HEIGHT = 280
LANDING_SPEED_LIMIT = -3

TIME_STEP = 0.001

t_podema = INITIAL_VELOCITY / GRAVITY
H_max = INITIAL_HEIGHT + INITIAL_VELOCITY * t_podema - 0.5 * GRAVITY * t_podema**2

v_i = -((2 * GRAVITY * (H_max - INITIAL_HEIGHT))**0.5)
delta_v = LANDING_SPEED_LIMIT - v_i
burn_duration = delta_v / (EXHAUST_VELOCITY * (FUEL_FLOW_RATE / (MASS_CRAFT + FUEL_INITIAL)) - GRAVITY)
fuel_required = FUEL_FLOW_RATE * burn_duration

H_s = H_max - ((abs(v_i + LANDING_SPEED_LIMIT))**2) / (2 * GRAVITY)
print(f"Входная скорость: {abs(v_i):.2f} м/с, Максимальная высота: {H_max:.2f} м, Высота включения двигателя: {H_s:.2f} м")

time_data = []
height_data = []
velocity_data = []
acceleration_data = []

current_time = 0
current_height = INITIAL_HEIGHT
current_velocity = INITIAL_VELOCITY
current_mass = MASS_CRAFT + FUEL_INITIAL
engine_active = False
remaining_fuel = FUEL_INITIAL

while current_height > -100:
    if not engine_active and current_height <= H_s:
        print(f"Двигатель включён на высоте {abs(current_height):.2f}м")
        engine_active = True

    if engine_active and remaining_fuel > 0:
        fuel_used = min(FUEL_FLOW_RATE * TIME_STEP, remaining_fuel)
        remaining_fuel -= fuel_used
        current_mass -= fuel_used

        thrust = EXHAUST_VELOCITY * FUEL_FLOW_RATE
        acceleration = thrust / current_mass - GRAVITY
    else:
        acceleration = -GRAVITY

    current_velocity += acceleration * TIME_STEP
    current_height += current_velocity * TIME_STEP

    time_data.append(current_time)
    height_data.append(max(0, current_height))
    velocity_data.append(current_velocity)
    acceleration_data.append(acceleration)

    current_time += TIME_STEP

    if current_height <= 0:
        break

time_data = np.array(time_data)
height_data = np.array(height_data)
velocity_data = np.array(velocity_data)
acceleration_data = np.array(acceleration_data)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time_data, height_data, label="Высота")
plt.title("Высота от времени")
plt.xlabel("Время, с")
plt.ylabel("Высота, м")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_data, velocity_data, label="Скорость", color="orange")
plt.title("Скорость от времени")
plt.xlabel("Время, с")
plt.ylabel("Скорость, м/с")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time_data, acceleration_data, label="Ускорение", color="green")
plt.title("Ускорение от времени")
plt.xlabel("Время, с")
plt.ylabel("Ускорение, м/с^2")
plt.grid(True)

plt.tight_layout()
plt.show()

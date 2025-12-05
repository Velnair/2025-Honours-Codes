# This example shows how to create a video of the Newton-Raphson method using
# matplotlib and numpy. The Newton-Raphson method is a root-finding algorithm
# that uses the derivative of a function to find its roots. The code generates
# an animation of the method applied to the function f(x) = x^3 - x - 2.
# Here we write the newton-raphson method by hand, rather than using scipy
# optimise methods which should be used in practice.

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Define the function and its derivative
def f(x):
    return x**3 - x - 2


def df(x):
    return 3 * x**2 - 1


# Newton-Raphson iteration
def newton_step(x):
    return x - f(x) / df(x)


# Generate points for animation
x_vals = [-1.5]  # initial guess
for _ in range(10):  # run for 10 steps
    x_vals.append(newton_step(x_vals[-1]))

# Set up the plot
fig, ax = plt.subplots()
x = np.linspace(-3, 3, 400)
y = f(x)
(line,) = ax.plot(x, y, label="f(x)")
(tangent_line,) = ax.plot([], [], "r--", label="Tangent")
(point,) = ax.plot([], [], "bo")
ax.axhline(0, color="black", lw=0.5)
ax.legend()
ax.set_title("Newton-Raphson Method")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")


# Animation function
def animate(i):
    # clear old lines
    for coll in ax.collections:
        coll.remove()
    if i == 0:
        point.set_data([], [])
        tangent_line.set_data([], [])
    else:
        x0 = x_vals[i - 1]
        y0 = f(x0)
        slope = df(x0)
        tangent_x = np.linspace(x0 - 0.5, x0 + 0.5, 100)
        tangent_y = slope * (tangent_x - x0) + y0
        tangent_line.set_data(tangent_x, tangent_y)
        point.set_data([x0], [y0])
    return line, tangent_line, point


ani = animation.FuncAnimation(
    fig, animate, frames=len(x_vals), interval=1000, blit=True
)

# Save the animation
output_path = "newton_raphson_animation.mp4"
# You will need to have ffmpeg installed on your machine to make a video
# it should be available on all systems through conda-forge
# https://anaconda.org/conda-forge/ffmpeg
ani.save(output_path, writer="ffmpeg", fps=1)

output_path

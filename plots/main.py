import matplotlib.pyplot as plt
import numpy as np

# Save figure
def save_fig(name):
    plt.savefig(name)

# Some example quadratic
def calc_z(x, y):
    return 0.25*x**2 + 2*y**2 + 0.5*x + 2*y + 4

# Plot the countours of z = x^2 + 2y^2 + 3x + 2y + 4
def plot_quadratic(n, x_range, y_range):
        x = np.linspace(*x_range, 100)
        y = np.linspace(*y_range, 100)

        x, y = np.meshgrid(x, y)

        z = calc_z(x, y) 
        
        fig = plt.figure()

        contour_plot = plt.contour(x, y, z, 10)

        plt.xlabel('x')
        plt.ylabel('y')

        fig.colorbar(contour_plot)

        plt.show()

if __name__=='__main__':
    # Generate various plots and save to file
    plot_quadratic(10, (-2.5, 0.5), (-1.25, 0.25))
    save_fig('quadratic.png')


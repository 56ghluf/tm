import matplotlib.pyplot as plt
from numpy import linspace, meshgrid, log, zeros, append, ones, array, nan

# Save figure
def save_fig(name):
    plt.savefig(name, bbox_inches='tight')

# Some example quadratic
def calc_z(x, y):
    return 0.25*x**2 + 2*y**2 + 0.5*x + 2*y + 4

# Plot the countours of z = x^2 + 2y^2 + 3x + 2y + 4
def plot_quadratic(n, x_range, y_range):
        x = linspace(*x_range, 100)
        y = linspace(*y_range, 100)

        x, y = meshgrid(x, y)

        z = calc_z(x, y) 
        
        fig = plt.figure()

        contour_plot = plt.contour(x, y, z, 10)

        plt.xlabel('x')
        plt.ylabel('y')

        fig.colorbar(contour_plot)

        plt.show()

def plot_log_vs_square():
    x = linspace(-0.1, 1.1, 100000)
    # Data for the -log
    y1 = -log(x)

    # Data for x^2
    y2 = x**2 
    
    # Create the subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Plot everything for log
    axs[0].plot(x, y1, color='tab:orange')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel(r'f(x)')
    axs[0].set_title(r'$f(x) = -\log(x)$')
    axs[0].set_xlim([-0.1, 1.1])
    axs[0].set_ylim([-0.2, 10])
    axs[0].set_aspect(0.1)
    
    # Plot everything for x^2
    axs[1].plot(x, y2, color='tab:red')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel(r'g(x)')
    axs[1].set_title(r'$g(x) = x^2$')
    axs[1].set_xlim([-0.1, 1.1])
    axs[1].set_ylim([-0.2, 10])
    axs[1].set_aspect(0.1)

    # plt.show()

def plot_relu_and_deriv():
    x1 = append(linspace(-5.2, 0, 3), linspace(0, 5.2, 3))
    y1 = append(zeros(3), linspace(0, 5, 3)) 

    x2 = append(linspace(-5.2, 0, 3), append(array([nan]), linspace(0.01, 5.2, 2)))
    y2 = append(zeros(3), ones(3)) 
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot relu
    axs[0].plot(x1, y1, color='tab:cyan')
    
    axs[0].set_title('$relu(x)$')

    axs[0].set_xlabel('x')
    axs[0].xaxis.set_label_coords(1, 0.47)

    axs[0].set_xlim([-5.2, 5.2])
    axs[0].set_ylim([-5.2, 5.2])

    axs[0].spines['left'].set_position('zero') 
    axs[0].spines['bottom'].set_position('zero') 
    axs[0].spines['right'].set_color('none')
    axs[0].spines['top'].set_color('none')

    axs[0].plot(1, 0, ">k", transform=axs[0].get_yaxis_transform(), clip_on=False)
    axs[0].plot(0, 1, "^k", transform=axs[0].get_xaxis_transform(), clip_on=False)

    axs[0].set_aspect(1)

    # Plot relu'
    axs[1].plot(x2, y2, color='tab:blue')

    axs[1].set_title('$relu\'(x)$')

    axs[1].set_xlabel('x')
    axs[1].xaxis.set_label_coords(1, 0.47)

    axs[1].set_xlim([-5.2, 5.2])
    axs[1].set_ylim([-5.2, 5.2])

    axs[1].spines['left'].set_position('zero') 
    axs[1].spines['bottom'].set_position('zero') 
    axs[1].spines['right'].set_color('none')
    axs[1].spines['top'].set_color('none')

    axs[1].plot(1, 0, ">k", transform=axs[1].get_yaxis_transform(), clip_on=False)
    axs[1].plot(0, 1, "^k", transform=axs[1].get_xaxis_transform(), clip_on=False)

    axs[1].set_aspect(1)

    # Plot the dicontinuity circles of relu'
    axs[1].scatter([0], [1], s=20, facecolors='tab:blue', edgecolors='tab:blue')
    axs[1].scatter([0], [0], s=20, facecolors='none', edgecolors='tab:blue')

    # plt.show()

if __name__=='__main__':
    # Generate various plots and save to file
    plot_relu_and_deriv()
    save_fig('ReluAndDeriv.png')

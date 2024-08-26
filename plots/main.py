import matplotlib.pyplot as plt
from numpy import linspace, meshgrid, log, zeros, append, ones, array, nan, vectorize, empty

# Save figure
def save_fig(name):
    plt.savefig(name, bbox_inches='tight')

def set_axs(axs):
    axs.spines['left'].set_position('zero') 
    axs.spines['bottom'].set_position('zero') 
    axs.spines['right'].set_color('none')
    axs.spines['top'].set_color('none')

    axs.plot(1, 0, ">k", transform=axs.get_yaxis_transform(), clip_on=False)
    axs.plot(0, 1, "^k", transform=axs.get_xaxis_transform(), clip_on=False)

# Some example quadratic
def calc_z(x, y):
    return 0.25*x**2 + 2*y**2 + 0.5*x + 2*y + 4

# Plot the countours of z = x^2 + 2y^2 + 3x + 2y + 4
def plot_quadratic(x_range, y_range):
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

def func1(x):
    a = lambda p: 0.5*(1.5*p**4 + p**3 - 4*p**2 + 3)
    a = vectorize(a)
    return a(x)

def deriv_func1(x):
    a = lambda p: 0.5*(6*x**3 + 3*p**2 - 8*p)
    a = vectorize(a)
    return a(x) 

def plot_momentum():
    x = linspace(-2.2, 2.2, 100)
    
    y = func1(x)
    
    # Run gradient descent without momentum  
    n1 = 100 
    preds = empty(n1+1)
    preds[0] = 1.75
    
    for i in range(n1):
        preds[i+1] = preds[i] - 0.01 * deriv_func1(preds[i])
    
    preds_y = func1(preds)

    # Run gradient descent with moementum
    n2 = 70 
    m_preds = empty(n2+1)
    m_preds[0] = 1.75

    m = deriv_func1(m_preds[0])
    
    beta = 0.95

    for i in range(n2):
        m = beta * m + (1-beta) * deriv_func1(m_preds[i])
        m_preds[i+1] = m_preds[i] - 0.01 * m

    m_preds_y = func1(m_preds)
    
    # Initialize the plots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # First plot
    axs[0].plot(x, y, color='tab:cyan', zorder=1)
    normal_plot = axs[0].quiver(
            preds[:-1], 
            preds_y[:-1],
            preds[1:]-preds[:-1],
            preds_y[1:]-preds_y[:-1],
            scale=1,
            scale_units='xy',
            angles='xy',
            color='tab:gray',
            zorder=2)

    axs[0].set_title(r'$f(x)$')

    axs[0].set_xlabel('x')
    axs[0].xaxis.set_label_coords(1, 0.39)

    axs[0].set_xlim(-2.2, 2.2)
    axs[0].set_ylim(-2.2, 3.2)

    set_axs(axs[0])

    # Second plot
    axs[1].plot(x, y, color='tab:cyan', zorder=1)
    momentum_plot = axs[1].quiver(
            m_preds[:-1], 
            m_preds_y[:-1],
            m_preds[1:]-m_preds[:-1],
            m_preds_y[1:]-m_preds_y[:-1],
            scale=1,
            scale_units='xy',
            angles='xy',
            color='k',
            zorder=2)

    axs[1].set_title(r'$f(x)$')

    axs[1].set_xlabel('x')
    axs[1].xaxis.set_label_coords(1, 0.39)

    axs[1].set_xlim(-2.2, 2.2)
    axs[1].set_ylim(-2.2, 3.2)

    set_axs(axs[1])

    fig.legend([normal_plot, momentum_plot], ['Optimisation sans élan', 'Optimisation avec élan'], loc=(0.8, 0.88))

    plt.show()

def plot_lin_sep():
    fx = [0, 1]
    fy = [1, 0]
    ex = [0, 1]
    ey = [0, 1]

    fig, axs = plt.subplots(1, figsize=(8, 8))

    axs.scatter(fx, fy, 150, zorder=3) 
    axs.scatter(ex, ey, 150, zorder=3)

    set_axs(axs)

    axs.set_xlabel('$p_1$')
    axs.set_ylabel('$p_2$')
    
    axs.set_xlim([-0.02, 1.05])
    axs.set_ylim([-0.02, 1.05])

    # plt.show()

def plot_lin_sep1():
    fx = [0, 1]
    fy = [1, 0]
    ex = [0, 1]
    ey = [0, 1]

    x = linspace(-1, 1, 2)
    y = -x + 0.8

    fig, axs = plt.subplots(1, figsize=(8, 8))

    axs.scatter(fx, fy, 150, zorder=3) 
    axs.scatter(ex, ey, 150, zorder=3)

    axs.plot(x, y, color='tab:cyan')

    set_axs(axs)

    axs.set_xlabel('$p_1$')
    axs.set_ylabel('$p_2$')
    
    axs.set_xlim([-0.02, 1.05])
    axs.set_ylim([-0.02, 1.05])

    # plt.show()

def plot_lin_sep2():
    fx = [0, 1]
    fy = [1, 0]
    ex = [0, 1]
    ey = [0, 1]

    x = linspace(-1, 2, 2)
    y = -x + 1.2 

    fig, axs = plt.subplots(1, figsize=(8, 8))

    axs.scatter(fx, fy, 150, zorder=3) 
    axs.scatter(ex, ey, 150, zorder=3)

    axs.plot(x, y, color='tab:cyan')

    set_axs(axs)

    axs.set_xlabel('$p_1$')
    axs.set_ylabel('$p_2$')
    
    axs.set_xlim([-0.02, 1.05])
    axs.set_ylim([-0.02, 1.05])

    # plt.show()

if __name__=='__main__':
    # Generate various plots and save to file
    plot_momentum()
    # save_fig('LinSep.png')

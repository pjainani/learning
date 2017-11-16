from numpy import *

def compute_error_for_given_points(b,m, points):
    totalError = 0.0
    for i in range(0, len(points)):
        x = points[i,0]
        y =points[i,1] # Check this step
        totalError +=(y-(m*x+b))**2
    return totalError/float(len(points))


def step_gradient(b_current, m_current, points, learning_rate):
    #todo
    b_gradient = 0 
    m_gradient = 0 
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += (2/N)*(y-(m_current*x + b_current))
        m_gradient += (2/N)*x* (y-(m_current*x + b_current))
    new_b = b_current - ( b_gradient*learning_rate)
    new_m=m_current - ( m_gradient*learning_rate)
    return [new_b,new_m]


def gradient_descent_runner(points,starting_b, starting_m, num_iterations,learning_rate):
    b = starting_b
    m = starting_m
    for  i in range(num_iterations):
        b,m = step_gradient(b,m,array(points),learning_rate)
    return [b,m]

def run():
    points = genfromtxt('data.csv', delimiter=',')
    #hyperparameters
    learning_rate = 0.0001

    #y=mx +b
    initial_b = 0
    initial_m = 0
    num_iterations  = 1000
    [b,m] = gradient_descent_runner(points,initial_b,initial_m,num_iterations,learning_rate)
    print(b)
    print(m)


if __name__ == '__main__':
    run()
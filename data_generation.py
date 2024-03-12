import numpy as np 
import scipy


def heat_exact(x: float, t: float, w: float, mu: float) -> float:
    '''
    Solution to the heat equation u_t-u_xx=0 with initial conditions u(0,t)=u(xr,t)=0
    '''
    return np.sin(w*x)*np.exp(-mu * w**2 * t)

def heat_for_testing(x: float, t: float, w: float, mu: float) -> float:
    '''
    Different solution to the heat equation u_t-u_xx=0 with initial conditions u(0,t)=u(xr,t)=0
    '''
    return np.sin(w*x)*np.exp(-mu * w**2 * t)+np.cos((w+1)*x)*np.exp(-mu * (w+1)**2 * t)

def initial_approx(func, xl=0, xr=2*np.pi, N_m=600, w=1, mu=10**-2, t=0):
    '''
    Finds average value of func wrt x in N_m cells from xl to xr at time t
    
    Parameters
    ----------
        func: {function}
            A function that takes the form func(x, t, w, mu)
        xl: float
            Left boundary
        xr: float
            Right boundary
        N_m: int
            Number of cells to integrate over
        w: float
            Solution parameter
        mu: float
            Solution parameter
        t: float
            time value
            
    Returns
    -------
    u_temp: numpy.ndarray
        Array of length N_m containing initial values of func at time t=0
    '''
    # Length of cells
    dx = (xr - xl) / N_m

    # ndarray of cell endpoints
    x = np.linspace(xl, xr, num=N_m+1)

    dt = 2*np.pi*dx
    
    # Solutions array
    u_temp = np.zeros(N_m)
    
    for loc in range(N_m):
        # Integrate u_exact wrt x and t=0, w=w, mu=mu
        u_temp[loc] = 1/dx * scipy.integrate.quad(func, x[loc], x[loc+1], args=(t,w,mu))[0]
        
    return u_temp

def extend_boundary(func, u_temp, xl=0, xr=2*np.pi, N_m=600, lb=5, rb=5, w=1, mu=10**-2, t=0):
    '''
    Extend initial solution u_temp beyond each boundary
    
    Parameters
    ----------
    func: {function}
        A function that takes the form func(x, t, w, mu)
    u_temp: numpy.array_like
        Array of average values of func of length N_m
    xl: float
        Left boundary
    xr: float
        Right boundary
    N_m: int
        Number of computational cells
    lb: float
        Amount to extend approximation to the left by
    rb: float
        Amount to extend approximation to the right by
    w: float
        Solution parameter
    mu: float
        Solution parameter
    t: float
        time
        
    Returns
    -------
    u_extended: numpy.ndarray
        Array which extends the approximation u_temp
    '''
    m = len(u_temp)
    
    # Length of cells
    dx = (xr - xl) / N_m
    
    u_extended = np.zeros(m+lb+rb)
    
    for loc in range(lb):
        # print(xl-dx*(lb-loc), " to ", xl-dx*(lb-(loc+1))) #For testing

        u_extended[loc] = 1/dx * scipy.integrate.quad(func, xl-dx*(lb-loc), xl-dx*(lb-(loc+1)), args=(t,w,mu))[0] #fixed -(loc+1) not -loc+1
        
    for loc in np.arange(lb,lb+m):
        u_extended[loc] = u_temp[loc-lb]
        
    for loc in np.arange(m+lb, m+lb+rb):
        u_extended[loc] = 1/dx * scipy.integrate.quad(func, xr +dx*(loc-m-lb), 2*xr+dx*(loc-m-lb+1), args=(t,w,mu))[0]
        
    return u_extended
"""
x=extend_boundary(heat_exact,initial_approx(heat_exact, w=5), w=5)
print(x)
print(extend_boundary(heat_exact,initial_approx(heat_exact, t=1, w= 5),t=1, w=5))

print(np.shape(x))
"""

# Generate data
# xl = 0
# xr = 2*np.pi
# N_m = 600
# dx = (xr - xl) / N_m
# lb = 5
# rb = 5
# dt = 2*np.pi*dx

def generate_data(func, frequency_array, xl=0, xr=2*np.pi, N_m=600, lb=5, rb=5,cfl_ratio = 2*np.pi):

    num_data_points = len(frequency_array)
    dx = (xr - xl) / N_m
    dt = cfl_ratio*dx
    X_intput_data = np.zeros((N_m+lb+rb,num_data_points))
    X_output_data = np.zeros((N_m,num_data_points))
    for i in range(num_data_points):
        w = frequency_array[i]
        '''
        #NON RESNET
        x_temp = extend_boundary(func,initial_approx(func, w=w), w=w, lb=lb, rb=rb)
        x_out_temp = initial_approx(func, w=w, t=dt)
        X_intput_data[:,i]=x_temp 
        X_output_data[:,i]=x_out_temp
        '''
        #RESNET
        x_time0 = initial_approx(func, w=w)
        x_temp = extend_boundary(func,x_time0, w=w, lb=lb, rb=rb)
        x_out_temp = initial_approx(func, w=w, t=dt)-x_time0
        X_intput_data[:,i]=x_temp 
        X_output_data[:,i]=x_out_temp

    
    return X_intput_data, X_output_data

print(np.shape(generate_data(heat_exact, np.arange(1,31,1))[1]))




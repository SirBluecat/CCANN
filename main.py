from model import *
from matplotlib import pyplot as plt

# Left boundary and right boundary
# Defines kernel size of CNN
left_boundary, right_boundary = 5, 5

# Training Parameters
learn_rate = .01 
learn_rate_decay = .85 #seemed to help training 
epochs = 5000
param_str = f"E{epochs}_LR{learn_rate}_LRD{learn_rate_decay}"

# Load or save model
load_model = True
save_model = False
model_location = "./saved_models/resnet_" + param_str



def testnetwork(func, model, loss, u_init,xL=0, xR=2*np.pi, N_m = 600, cfl = 2*np.pi, T_final = 1):
    '''
    Testing function which generates ground truth data to test against
    
    Parameters
    ----------
        func: {function}
            Function which is being learned by CCANN
        model: 
            CCANN which has been trained on func
    '''
    dx = (xR-xL)/N_m
    dt = cfl*dx
    T_final = 1
    x = np.linspace(xL, xR, num=N_m+1) # x domain
    xx = np.linspace(xL+dx/2,xR-dx/2, num=N_m) # tweaked x domain

    t= np.arange(0,T_final,dt) # time range
    N_t = len(t)
    
    u_init_tensor = torch.from_numpy(u_init.astype(np.float32)).to(device)
    u_extend = torch.from_numpy(extend_boundary(func=func, u_temp = u_init, t=t[0]).astype(np.float32)).to(device)
    #u_extend = u_extend.reshape((1,610))
    #send to tensor
    
    u_network_approx = torch.zeros(N_m, N_t) # Array whose column t is the model approx of time dt>0
    u_exact = torch.zeros(N_m,N_t) # Array whose column t is the ground truth of time dt>0
    u_network_approx[:,0] = u_init_tensor
    u_exact[:,0] = u_init_tensor

    for i in range(N_t-1):
        u_extend = u_extend.reshape((1,610))
        # u_network_approx[:,i+1]=model(u_extend) # Predicts next timestep into the future NON RESNET
        u_network_approx[:,i+1] = u_network_approx[:,i]+model(u_extend) #RESNET
        u_extend = torch.from_numpy(extend_boundary(func=func, u_temp = u_network_approx[:,i+1], t = t[i+1]).astype(np.float32)).to(device)
        exact_soln_temp = initial_approx(func=func, t=t[i+1])
        u_exact[:,i+1]=torch.from_numpy(exact_soln_temp.astype(np.float32)).to(device)
        #send extend_boundary to tensor

    L2_error = loss(u_exact,u_network_approx)
    print(L2_error)
    plt.scatter(xx,u_network_approx[:,N_t-1].detach().numpy()) # cast to numpy
    plt.scatter(xx,u_exact[:,N_t-1].detach().numpy())
    plt.show()
    



if __name__ == '__main__':
    criterion = nn.MSELoss()
    
    if load_model == False:
        # Instantiate network as model
        # Kernel size is lb + rb + 1   
        model = get_network(left_boundary, right_boundary)
        
        
        X_input=generate_data(heat_exact, np.arange(1,31,1), lb=left_boundary, rb=right_boundary)[0] 
        X_output = generate_data(heat_exact, np.arange(1,31,1), lb=left_boundary, rb=right_boundary)[1]

        print("In vector" + str(X_input.shape))
        print("Out vector" +str(X_output.shape))
        
        X_input = (X_input.T).reshape((np.shape(X_input)[1],1,np.shape(X_input)[0]))
        X_output = (X_output.T).reshape((np.shape(X_output)[1],1,np.shape(X_output)[0]))
        

        
        # X_input=(X_input.T).reshape((np.shape(X_input)[1],1,np.shape(X_input)[0]))
        # X_output = (X_output.T).reshape((np.shape(X_output)[1],1,np.shape(X_output)[0]))
        
        X_in_tensor = torch.from_numpy(X_input.astype(np.float32)).to(device)
        X_out_tensor = torch.from_numpy(X_output.astype(np.float32)).to(device)
        
        
        
        print("In tensor " + str(X_in_tensor.size()))
        print("Out tensor " + str(X_out_tensor.size()))
        
        ########### Training #############
        
        optimizer = get_optim(model, learn_rate=learn_rate)
        losses = train_net(model, criterion, learn_rate, optimizer, learn_rate_decay, epochs, x=X_in_tensor, y= X_out_tensor)
        itter_number = np.arange(0,epochs)
        # plt.plot(itter_number, losses)
        # plt.show()
        print("Trained.")
        
        if save_model == True:
            torch.save(model, model_location)
    else:
        model = torch.load(model_location)
    
    ######### Testing ############

    u = initial_approx(heat_for_testing)
    # u = inital_approx(heat_exact, w= 13.7)

    testnetwork(heat_for_testing, model, criterion, u)
    
    
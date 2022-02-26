#%%
#%%
import pandas as pd
import numpy as np
import random
#%%
os.chdir('/Users/patrickkampmeyer/Dropbox/Ph.D/Classes/Spring_2022/Machine_Learning/Project_2/dataset')

data = pd.read_csv('blobs.csv', index_col = 0)

#%%


X = data.iloc[:, :-1].values
Y_1 = data.iloc[:, -1].values.reshape(-1,1)
Y = np.where(Y_1 == 0, -1, 1)

#%%
def linear(x,z):
    return np.dot(x, z.T)

# %%
def gaussian(x,z, sigma=0.1): 
    return np.exp(-np.linalg.norm(x-z, axis=1)**2 / (2*(sigma**2)))
# %%

kernal = linear

m,n = X.shape

K = np.zeros((m,m))

for i in range(m):
    # inputs for the kernal are (x,z).. the variables we
    # want to transform
    K[i,:] = kernal(X[i,np.newaxis], X)
# %%

# what do we need to actually predict? 

y_predict = np.zeros((X.shape[0]))

sv = get_parameters(alphas)
#%%


# need to first provide alphas.. get the support vector elements (xis) for which the alphas are within range

def get_parameters(alphas):

        # setting the support vectors

        C = 1
        threshold = 1e-4
        sv = ((alphas > threshold) * (alphas < C)).flatten()
        w = np.dot(X[sv].T, alphas[sv]*y[sv, np.newaxis])
        b = np.mean(y[sv, np.newaxis] -
                            alphas[sv]*y[sv, np.newaxis]*K[sv,sv][:,np.newaxis])


# NEED TO GET THE ALPHAS FOR THIS TO WORK

#%%
# *********************************************** initialize passes **********************************************
passes = 0 
tol = 0.30
max_passes = 2
C = 10

num_instances = len(X)
alpha = np.ones((num_instances,1)) #* 0.1
alpha_old = np.ones((num_instances,1)) #* 0.1

b = np.zeros((num_instances,1))


#%%
# begin the loop

# compute the kernal for all instances
kernal_i = np.expand_dims(K.sum(axis=1), axis=1)

# compute f(x)
f_x = (alpha * Y * kernal_i) + b
E =  f_x - Y


while (passes < max_passes):

    print("pass is", passes)

    num_changed_alphas = 0
    
    
    for i in range(1,num_instances):

        print("at next instance", i)

       #Calculate E_i = f(xi)-yi using (2)
        f_x = (alpha * Y * kernal_i) + b
        E[i] =  f_x[i] - Y[i]

        # Checking to see if we're in the bounds

        if ((Y[i] * E[i] < -tol) and (alpha[i] < C)) or ((Y[i] * E[i] > tol) and (alpha[i] > 0)): 

            
            # select i /= j randomly
            j = random.choice([j for j in range(1,num_instances) if j != i])

            print("outside tol 1.. pick j=", j)

            # Calculate Ej = f(xj)-yj using (2)
            E[j] =  f_x[j] - Y[j]


            # Save old alphas 
            alpha_old[i] = alpha[i]

            alpha_old[j] = alpha[j]

            # Compute L and H by (10) and (11)

            if (Y[i] == Y[j]):
                
                L = max(0, alpha[j] - alpha[i])
                H = min(C, C + alpha[j] - alpha[i])
        
            else:

                L = max(0, alpha[i] + alpha[j]- C)
                H = min(C,alpha[i] + alpha[j])

            if (L==H):
                print("bounds are equal.. continue to next instance")
                continue


            # compute n by (14)
            eta = 2*(np.dot(X[i],X[j]))-np.dot(X[i],X[i])-np.dot(X[j],X[j])

            if (eta >= 0):
                print("eta greater than zero.. continue to next instance")
                continue

            # compute and clip new value for alphaj using (12) and (15)

            # 12

            alpha[j] = alpha[j] - (Y[j]*(E[i] - E[j])/eta)


            # 15

            if(alpha[j] > H): 
                alpha[j] = H

            if (alpha[j]>= L) and (alpha[j] <= H):
                alpha[j] = alpha[j]

            else:
                alpha[j] = L

                            
            if (abs(alpha[j]-alpha_old[j]) < 10e-5):
                print("alphas dont change.. continue to next instance")
                continue

            # Determine value of alphi_i using (16)
            alpha[i] = alpha[i]+ ((Y[i] * Y[j])*(alpha_old[j] - alpha[j]))

        # Compute b1 and b2 using (17) and (18) respectively. 

            b1 = b - E[i]-(Y[i]*(alpha[i] - alpha_old[i])*np.dot(X[i],X[i])) - (Y[j]*(alpha[j] - alpha_old[j])*np.dot(X[i],X[j]))
                            
            b2 = b - E[j]-(Y[i]*(alpha[i] - alpha_old[i])*np.dot(X[i],X[j])) - (Y[j]*(alpha[j]  - alpha_old[j])*np.dot(X[j],X[j]))
        
        # Compute b by (19)

            if (alpha[i] > 0) and (alpha[i] < C):
                                
                b = b1

            if (alpha[j] > 0) and (alpha[j] < C):

                b = b2

            else: 
                                
                b = (b1 + b2)/2

            num_changed_alphas = num_changed_alphas + 1

    if (num_changed_alphas == 0):
        passes = passes + 1 
    else:
        passes = 0


#%%

y_predict = np.zeros((X.shape[0]))

sv = get_parameters(alpha)


for i in range(X.shape[0]):
    Y_predict[i] = np.sum(alpha[sv] * Y[sv, np.newaxis]*
                                    kernal(X[i], X[sv])[:,np.newaxis])

    return np.sign(Y_predict + b)

#%%  

#def get_parameters(self, alphas):

    # setting the support vectors

threshold = 1e-4
sv = ((alpha > threshold) * (alpha < C)).flatten()
w = np.dot(X[sv].T, alpha[sv]*Y[sv, np.newaxis])
b = np.mean(Y[sv, np.newaxis] -
                    alpha[sv]*Y[sv, np.newaxis]*K[sv,sv][:,np.newaxis])

# %%

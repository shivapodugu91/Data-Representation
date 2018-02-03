
# coding: utf-8

# In[1]:


import sys
import numpy as np
if len(sys.argv) != 3:
    print(sys.argv[0], " takes 2 arguments .\"dataFileName, labelsFileName\". Not ", len(sys.argv) - 1)
    sys.exit()
dataFileName = sys.argv[1]
print(dataFileName)
# In[13]:


data = np . genfromtxt (dataFileName, delimiter =',') # default delimiter is space
#print ("data:\n", data )
X = np.matrix(data)
#X


# In[18]:


#mean_rows = np. mean (X, axis =0); print (" mean_rows =" , mean_rows )
mean_cols = np. mean (X, axis =1); print (" mean_cols :\n" , mean_cols )
X = X-mean_cols


# In[19]:


X


# In[20]:


#covariance X*Xt
covariance = np.dot(X.T,X);
#covariance = np.cov(X, rowvar=False)
print ("covariance=\n" , covariance)



# In[21]:


np.array_equal(covariance,covariance.T)


# In[22]:


if(np.array_equal(covariance,covariance.T)):
    print("the covariance matrix is symmetric\n")
    evals , evecs = np . linalg . eigh( covariance );
    print (" eigen_values =", evals , "\n eigen_vectors:\n", evecs )
else:
    print("the covariance matrix is not symmetric\n")
    A = np.zeros(covariance.shape)
    A = A+1
    C = np . dot ( np . linalg . pinv ( covariance ) , A );
    evals , evecs = np . linalg . eig ( C );
    print (" eigen_values =", evals , "\n eigen_vectors:\n", evecs )
    
    


# In[23]:


# eigenvalues in increasing order , not decreasing order . Sort them .
idx = np . argsort ( evals )[:: -1] # sort in reverse order
evals = evals [ idx ]
evecs = evecs [: , idx ]
print ("evals:\n", evals , " \n\nevecs:\n", evecs ) # evectors are the cols of evecs
# extract the 2 dominant eigenvectors
r = 2
V_r = evecs [: ,: r ]; print ("\nV_r:\n", V_r ) # get first r eigenvectors


# In[24]:


#projections
p=X*V_r
p


# In[10]:
outputfilename = sys.argv[1]+'.pca2_output'

np.savetxt(outputfilename, p)


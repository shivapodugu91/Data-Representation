
# coding: utf-8

# In[12]:

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




# In[14]:


X = np.matrix(data)
#X


# In[15]:


#covariance = Xc.T*Xc; print ("covariance=" ,covariance)


# In[16]:


#covariance = np.cov(X, rowvar=False)


# In[17]:


covariance = np.dot(X.T,X);
print ("covariance=\n" , covariance)


# In[18]:


np.array_equal(covariance,covariance.T)


# In[19]:


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
    
    


# In[20]:


# eigenvalues in decreasing order . Sort them .
idx = np . argsort ( evals )[:: -1] # sort in reverse order
evals = evals [ idx ]
evecs = evecs [: , idx ]
print ("evals:\n", evals , " \n\nevecs:\n", evecs ) # evectors are the cols of evecs
# extract the 2 dominant eigenvectors
r = 2
V_r = evecs [: ,: r ]; print ("\nV_r:\n", V_r ) # get first r eigenvectors


# In[21]:


#projections
p=X*V_r
p


# In[22]:
outputfilename = sys.argv[1]+'.pca1_output'

np.savetxt(outputfilename, p)


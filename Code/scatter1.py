
# coding: utf-8

# In[ ]:


#minimize the within-class scatter


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

# In[3]:


y = np . genfromtxt (sys.argv[2]) # default delimiter is space
#print ("y:\n", y )
Y =np.matrix(y)


# In[4]:


X = np.matrix(data)


# In[5]:


m,cols= (X.shape)


# In[6]:


#u
mean_rows = np. mean (X, axis =0); 
print (" mean_rows =" , mean_rows )


# In[7]:


my_dict = {} # count of each unique labels
my_dict2 = {} # sum of data corresponding to unique labels
temp=0
for key in y:
#print(key in my_dict)
    if((key in my_dict)==0):
        my_dict[key]=1
        my_dict2[key]= X[temp]
        temp = temp+1
    else:
        my_dict[key]=my_dict[key]+1
        my_dict2[key]= my_dict2[key] + X[temp]
        temp = temp+1


# In[8]:


my_dict2


# In[9]:


# calculating mean of data corresponding to unique labels
for key in my_dict2:
    my_dict2[key] = my_dict2[key]/ my_dict[key]


# In[10]:


my_dict2


# In[11]:


u_j=np.zeros((cols,m))
#u[0]+my_dict2[1.0].T
u_j.shape


# In[12]:


temp=0
for key in y:
    u_j[:,temp]=u_j[:,temp]+my_dict2[key]
    temp = temp+1
u_j


# In[13]:


#W = np.dot(X.T-u_j,(X.T-u_j).T)
#W


# In[15]:


#covariance=W


# In[16]:


#covariance X*Xt

covariance = np.cov((X.T-u_j).T, rowvar=False)

print ("covariance=\n" , covariance)



# In[17]:


np.array_equal(covariance,covariance.T)


# In[18]:


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
    
    


# In[19]:


# eigenvalues in increasing order
idx = np . argsort ( evals )[:: 1] 
evals = evals [ idx ]
evecs = evecs [: , idx ]
print ("evals:\n", evals , " \n\nevecs:\n", evecs ) # evectors are the cols of evecs
# extract the 2 dominant eigenvectors
r = 2
V_r = evecs [: ,: r ]; print ("\nV_r:\n", V_r ) # get first r eigenvectors


# In[ ]:


# Compute the projections of Xt on v1 , v2 for the matrix D of size nx2


# In[20]:


#projections
p=X*V_r
p


# In[21]:

outputfilename = sys.argv[1]+'.scatter1_output'

np.savetxt(outputfilename, p)



# In[ ]:





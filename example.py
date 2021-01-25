import matplotlib.pyplot as plt
import numpy as np
import PurePythonNN
# training data same as test data
X=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
y=X

net = PurePythonNN.netWork()
net.Learn(50000,3000,0.1,X,y,PurePythonNN.MomentumDesent)
#Produces error sum plot
plt.scatter(net.listCycle,net.listError)
plt.show()

# Pure-python-Neural-Network
a pure Python Neural Network with configurable size and back propagation algorithm. Currently implements Classic gradient descent, a modified RMSprop( no epsilon calculated) and Momentum

## PurePythonNN
```python
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

# 50000 iterations, logs every 3000, 0.1 LearnRate. input data and Output data. DesentMethod (respectivly) 
```
> example use case: example.py

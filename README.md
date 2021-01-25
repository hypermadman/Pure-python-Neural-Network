# Pure-python-Neural-Network
a pure Python Neural Network with configurable size and back propagation algorithm. Currently implements Classic gradient descent, a modified RMSprop( no epsilon calculated) and Momentum

## PurePythonNN
```python
import PPurePythonNN
net = PurePythonNN.netWork()
net.Learn(50000,3000,0.1,X,y,RMSDesent)
# 50000 iterations, logs every 3000, 0.1 LearnRate. input data and Output data. DesentMethod (respectivly) 
```

import numpy as np
import matplotlib.pyplot as plt

def Sigmoid(x):
    return 1/(1+np.exp(-x))
def apxInvSigmoid(x):
    return x*(1-x)

#X = np.array([[0.01,0.99,0.01,0.01],[0.99,0.01,0.01,0.01],[0.01,0.01,0.99,0.01],[0.01,0.01,0.01,0.99]])
X=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
y=X

np.random.seed(1)
class netWork():
    def __init__(self):
# randomly initialize our weights with mean 0
        self.syn0 = 4*(0.75**.5)*( 2*np.random.random((4,2)) - 1 )
        self.syn1 = 4*(0.75**0.5)*( 2*np.random.random((2,4)) - 1 )
        # doing rearch having lots of issues with paths though ending up identical this problem is offten called symetry and getting arround it is called symmetry breaking
        # the main way to get arround this is to change how the weights are initialized as to give better spread this should reduce/ remove this problem
        # Xavier initialization is one of these methods
        # times all by (2/ l-1)**.5
        # another is (2/(l+ l+1))**0.5
        # uniform xavier another (6/l + l-1)**0.5
        #xavier-glorot is met to be best for unifrom with sigmoid function
        # (6/in+out)**0.5
        self.Cyclecount =0
        self.l2_Sum =0
        self.l1_Sum =0
        self.listCycle =[]
        self.listError =[]
    def Learn(self,runs,outmod,LearnRate,inputs,outputs):
        for j in range(runs):
            # Feed forward through layers 0, 1, and 2
            l0 = inputs
            l1 = Sigmoid(l0@self.syn0)
            l2 = Sigmoid(l1@self.syn1)
            # how much did we miss the target value?
            l2_error = outputs - l2
            l2_delta = l2_error*apxInvSigmoid(l2)
            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            # how much did each l1 value contribute to the l2 error (according to the weights)?
            l1_error = l2_delta.dot(self.syn1.T)
            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            l1_delta = l1_error * apxInvSigmoid(l1)
            l2_Grad = l1.T.dot(l2_delta)
            l1_Grad = l0.T.dot(l1_delta)
            #self.l2_Sum = self.l2_Sum*0.6 - l2_Grad
            #self.l1_Sum = self.l1_Sum*0.6 - l1_Grad
            self.l2_Sum = self.l2_Sum*0.9 + 0.1* l2_Grad**2
            self.l1_Sum = self.l1_Sum*0.9 + 0.1* l1_Grad**2
            #print(self.l2_Sum,self.l1_Sum)

            self.syn1 +=  (( LearnRate )* self.l2_Sum**-0.5  )* l2_Grad
            self.syn0 +=  (( LearnRate )* self.l1_Sum**-0.5  )* l1_Grad
            #self.syn1 += LearnRate*l1.T.dot( l2_delta)
            #self.syn0 += LearnRate*l0.T.dot( l1_delta)
            #self.syn1 -= LearnRate*self.l2_Sum
            #self.syn0 -= LearnRate*self.l1_Sum
            if ((self.Cyclecount+j)% outmod) == 0:
                #error =np.mean(np.abs(l2_error))
                error =np.sum(np.abs(l2_error))
                print ( "Error:" + str(error))
                #print((self.syn1))
                #print(self.syn0)
                print(l2)
                #print(l2_error)
                self.listCycle.append(self.Cyclecount+j)
                self.listError.append(error)
        self.Cyclecount +=runs
    def IdentifyWorstTests(self,inputs,outputs,ErrorLim):
        l0 = inputs
        l1 = Sigmoid(l0@self.syn0)
        l2 = Sigmoid(l1@self.syn1)
        # how much did we miss the target value?
        l2_error = outputs - l2
        print(np.sum(np.abs( l2_error ),axis = 0))
        return [ num for num ,ResultError in enumerate( np.sum(np.abs( l2_error ),axis = 0) ) if ResultError>ErrorLim ]
net = netWork()

net.Learn(50000,3000,0.1,X,y)

plt.scatter(net.listCycle,net.listError)
            #print(syn1,syn0)
plt.show()
# an option which may get arround my predicament is to use PID esk formula adding I (historic error) we may be able to prevent landing in local minima or adding D inter step gradent (this is often called momentum)
# changing np.dot too @ for speed gains

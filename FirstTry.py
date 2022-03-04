from mimetypes import init
import numpy as np

# Generate training data
def f():
    a = int(np.random.uniform(0, 50))
    b = int(np.random.uniform(0, 50))

    return [a, b, a+b]

#X-train is [:2], Y-train is[2]
a = np.array([f() for i in range(10000)])
val = np.array([f() for i in range(10^3)])

# initialize initial(random) weights from layer m to layer n, normalize so add to 1
def initWeights(m_param, n_param):
    return np.random.uniform(-1, 1, size=(m_param, n_param)) / np.sqrt(m_param*n_param)

l1 = initWeights(50,75)
l2 = initWeights(75,100)


def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)

def softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)

def d_softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))

#forward and backward pass
def forward_backward_pass(x,y):
    targets = np.zeros((len(y),100), np.float32)
    targets[range(targets.shape[0]),y] = 1


    x_l1=x.dot(l1)
    x_sigmoid=sigmoid(x_l1)

    print(x_sigmoid.shape)
    x_l2=x_sigmoid@l2
    out=softmax(x_l2)


    error=(2*(out-targets)/out.shape[0]*d_softmax(x_l2))
    update_l2=x_sigmoid.T@error
    
    
    error=((l2).dot(error.T)).T*d_sigmoid(x_l1)
    update_l1=x.T@error

    return out,update_l1,update_l2 

epochs=10000
lr=0.001
batch=128

losses,accuracies,val_accuracies=[],[],[]


for i in range(epochs):
    y = [a[i][2]]
    x = np.zeros((2,50), np.int32)
    x[0][a[i][0]] = 1
    x[1][a[i][1]] = 1
    print(x)


    out,update_l1,update_l2=forward_backward_pass(x,y)
    
    category=np.argmax(out,axis=1)
    accuracy=(category==y).mean()
    accuracies.append(accuracy)

    loss=((category-y)**2).mean()
    losses.append(loss.item())

    l1=l1-lr*update_l1
    l2=l2-lr*update_l2

    if(i%20==0):    
        X_val = np.zeros((2,50), np.int32)
        X_val[i][a[i][0]] = 1
        X_val[i][a[i][1]] = 1
        val_out=np.argmax(softmax(sigmoid(X_val.dot(l1)).dot(l2)),axis=1)
        print(val_out)
        val_acc=(val_out==a[i][2]).mean()
        val_accuracies.append(val_acc.item())
        if(i%500==0): print(f'For {i}th epoch: train accuracy: {accuracy:.3f} | validation accuracy:{val_acc:.3f}')
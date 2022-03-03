from mimetypes import init
import numpy as np

# Generate training data
def f():
    a = int(np.random.uniform(0, 50))
    b = int(np.random.uniform(0, 50))

    return [a, b, a+b]

a = np.array([f() for i in range(10)])

# initialize initial(random) weights from layer m to layer n, normalize so add to 1
def initWeights(m_param, n_param):
    return np.random.uniform(-1, 1, size=(m_param, n_param)) / np.sqrt(m_param*n_param)

l1 = initWeights(100, 10)
l2 = initWeights(10, 100)


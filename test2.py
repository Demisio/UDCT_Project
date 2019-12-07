import numpy as np


def sample(x):
    print(np.random.choice(x))

def shuffle(x):
    np.random.shuffle(x)
    print(x)

a = np.array([1,2,2,1,5,5])
print(np.count_nonzero(a == 4))
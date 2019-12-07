import numpy as np


def sample(x):
    print(np.random.choice(x))

def shuffle(x):
    np.random.shuffle(x)
    print(x)

# a = np.array([1,2,2,1,5,5])
# print(np.count_nonzero(a == 4))

a = np.array([(1,3,5), (7,9,0),(1,3,5)]).reshape((3,3,1))
b = np.array([(2,4,6), (8,0,2),(2,4,6)]).reshape((3,3,1))

# b = 2
# c = 3

print(a.flatten())
print(b.flatten())
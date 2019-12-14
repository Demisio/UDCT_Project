import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
def sample(x):
    print(np.random.choice(x))

def shuffle(x):
    np.random.shuffle(x)
    print(x)

# a = np.array([1,2,2,1,5,5])
# print(np.count_nonzero(a == 4))

# a = np.array([(1,3,5), (7,9,0),(1,3,5)]).reshape((3,3,1))
# b = np.array([(2,4,6), (8,0,2),(2,4,6)]).reshape((3,3,1))

# b = 2
# c = 3
#
# print(a.flatten())
# print(b.flatten())


# a = np.arange(0, 200)
# b = np.arange(200,400)
#
# fig, axes = plt.subplots(1,2)
#
# axes[0].plot(np.arange(200),a)
# axes[1].plot(np.arange(200),b)
# # plt.show(block=False)
# plt.ion()
# plt.show()
# plt.pause(0.001)
#
# for i in range(1000000):
#     print(i)



a = np.array([0.035,0.040,0.038])
b = np.array([0.038, 0.038, 0.039])

corr, _ = pearsonr(a,b)
print(corr)
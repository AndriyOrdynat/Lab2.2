import numpy as np
import random
import matplotlib.pyplot as plt

# 1

# M1 = np.random.randn(25,25)
# print(M1.max())
# print(M1.min())

# M2 = np.diag([1,2,3,4,5,0])
# print(M2)

# M4 = np.random.randint(0,9, (5,3))
# M5 = np.random.randint(0,9, (3,2))
# M45 = np.dot(M4,M5)
# print(M4,'\n \n', M5, '\n')
# print(M45)

# ar1 = np.array(range(20))
# ar1[9:15] *= -1
# print(ar1)

# 2.1
# def create_original_Y(X):
#     Y = np.random.randint(1, 11, len(X))
#     while True:
#         if len(np.intersect1d(X, Y)) == 0:
#             break
#         for yi in range(len(Y)):
#             while Y[yi] in X:
#                 Y[yi] = np.random.randint(1, 11)
#     return Y

                
# X = np.random.randint(1, 11, 5)
# Y = create_original_Y(X)
            
            
# print(X, '----' ,Y)

# def Koshi_matrix(X, Y):
#     return 1/np.subtract.outer(X, Y)

# print(Koshi_matrix(X, Y))
# print("Детермінант матриці коші :", np.linalg.det(Koshi_matrix(X, Y)))

# 2.2

# def find_extrem(func , a, b, n):
#     x = np.linspace(a, b, n)
#     y = func(x) 
#     max_index = np.argmax(y)
#     min_index = np.argmin(y)
#     return x[max_index], y[max_index], x[min_index], y[min_index]

# def f(x):
#     return x**2 

# extr = find_extrem(f, -10, 10, 1000)
# print(f'Максимум: ({extr[0]}, {extr[1]})')
# print(f'Мінімум: ({extr[2]}, {extr[3]})')

# 3

x = np.arange(0, 5.1, 0.01)
y_el = np.zeros(len(x))
y_el[0] = 1
for i in range(1, len(x)):
    y_el[i] = y_el[i-1] + 0.01 * (y_el[i-1]*(1-2*x[i-1]))
# print(y_el)

y_an = np.exp(x-x**2)
# print(y_an)

plt.plot(x, y_el)
plt.plot(x, y_an, linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("y' = y(1 - 2x), y(0)=1")
plt.legend()
plt.show()
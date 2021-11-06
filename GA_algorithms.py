# !/usr/bin/python
# -*- coding:utf-8 -*-

# pip install scikit-opt

# 遗传算法
# 定义问题
from sko.PSO import PSO
from sko.DE import DE
import matplotlib.pyplot as plt
import pandas as pd
from sko.GA import GA
import numpy as np


def obj_func(p):
    x1, x2, x3 = p
    return x1 ** 2 + x2 ** 2 + x3 ** 2


if __name__ == "__main__":
    print(obj_func([1, 2, 3]))


def schaffer(p):
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.sin(x) - 0.5) / np.square(1 + 0.001 * x)


# 运行遗传算法
ga = GA(func=schaffer, n_dim=2, size_pop=50, max_iter=800,
        lb=[-1, -1], ub=[1, 1], precision=1e-7)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

# 用matplotlib画出结果

Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()

#################################
# 差分进化
'''
min f(x1, x2, x3) = x1^2 + x2^2 + x3^2
s.t.
    x1*x2 >= 1
    x1*x2 <= 5
    x2 + x3 = 1
    0 <= x1, x2, x3 <= 5
'''

# 目标函数


def obj_func(p):
    x1, x2, x3 = p
    return x1 ** 2 + x2 ** 2 + x3 ** 2


# 线性约束
constraint_eq = [
    lambda x: 1 - x[1] - x[2]
]

# 非线性约束
constraint_ueq = [
    lambda x: 1 - x[0] * x[1],
    lambda x: x[0] * x[1] - 5
]

# 进行差分进化
de = DE(func=obj_func, n_dim=3, size_pop=50, max_iter=800, lb=[0, 0, 0], ub=[
        5, 5, 5], constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)
best_x, best_y = de.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)


################################################
# 粒子群算法
# pso = PSO(func=demo_func, dim=3, pop=40, max_iter=150, lb=[0, -1, 0.5], ub=[1, 1, 1], w=0.8, c1=0.5, c2=0.5)
# pop: 类型int，种群的大小，也就是粒子的数量。我们使用"pop"来与GA保持一致。默认40
# lb: 类型列表，下限。每个参数的下限
# ub: 类型列表，上限。每个参数的上限

# 有限制粒子群
# 第一步，定义问题
def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2


# 第二步
pso = PSO(func=demo_func, dim=3, pop=40, max_iter=150, lb=[
          0, -1, -0.5], ub=[1, 1, 1], w=0.8, c1=0.5, c2=0.5)
pso.run()
print("粒子群算法运行结果:")
# gbest_x: array_like, shaoe is (1, dim) general best location for all particles in history
# gbest_y: float general best image for all particls in history
# gbest_y_hist: list gbest_y of every iteration
print('best_x is ', pso.gbest_x, 'best_y is ', pso.gbest_y)
print(pso.gbest_y_hist)

# 画出结果
plt.plot(pso.gbest_y_hist)
plt.show()

# 无限制粒子群算法
# %%定义函数


def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2


# %% Do PSO

pso = PSO(func=demo_func, dim=3)
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

# %% Plot the result

plt.plot(pso.gbest_y_hist)
plt.show()


# %%数据批量做粒子群优化
def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2


data = {'lb': [[0, -1, 0.5], [1, 1, 1], [2, 3, 4]],
        'ub': [[1, 1, 1], [2, 2, 2], [4, 5, 6]]}
data = pd.DataFrame(data)


print(data.shape[0])


def pso(lb, ub):
    pso = PSO(
        func=demo_func,
        dim=3,
        pop=40,
        max_iter=150,
        lb=lb,
        ub=ub,
        w=0.8,
        c1=0.5,
        c2=0.5)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)


for i in range(data.shape[0]):
    pso(data['lb'][i], data['ub'][i])



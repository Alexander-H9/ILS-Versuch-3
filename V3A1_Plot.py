import matplotlib.pyplot as plt

pX1 = [-2], [-2], [-1.5], [0], [2], [3], [4], [4]
pY1 = [1, 2, 1, 2, 1, 0,-1,2]
pX2 = [-1],[-0.5],[0,],[0.5],[1],[2],[3]
pY2 = [-2],[-1],[0.5],[-2],[0.5],[-1],[-2]

plt.scatter(pX1, pY1)
plt.scatter(pX2, pY2)

plt.show()
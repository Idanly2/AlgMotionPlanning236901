import matplotlib.pyplot as plt

dimensions = [i for i in range(2, 11)]
epsilons = [0.2, 0.1, 0.01]

fig = plt.figure()
ax1 = fig.add_subplot(111)

for eps in epsilons:
    eta = []
    for d in dimensions:
        eta.append(1 - (1 - eps) ** d)
    ax1.scatter(dimensions, eta, s=10, marker="s", label='$\epsilon$ = ' + str(eps))
plt.legend(loc='upper left')
plt.title('The fraction of the volume that is ε distance from the surface of a d-dimensional unit ball')
plt.xlabel('d')
plt.ylabel('${\eta_{d}({\epsilon})}$')
plt.grid()
plt.show()

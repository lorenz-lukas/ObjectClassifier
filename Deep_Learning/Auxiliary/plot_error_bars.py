import matplotlib.pyplot as plt
import numpy as np

x = np.array([3, 5, 10, 15, 20, 30],dtype=np.int32)
y = 100*np.array([0.191780, 0.247973, 0.392447, 0.456535, 0.536555, 0.645337])
e = 100*np.array([0.176787, 0.199224, 0.225419, 0.218645, 0.221468, 0.223282])

plt.errorbar(x, y, e, linestyle='None', marker='^')

plt.ylabel('Porcentagem (%)')
plt.xlabel('# de exemplos de treino por classe')

plt.savefig('triplet_results.png', bbox_inches='tight')
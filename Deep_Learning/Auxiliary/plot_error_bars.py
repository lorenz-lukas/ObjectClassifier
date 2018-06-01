import matplotlib.pyplot as plt
import numpy as np

x = np.array([3, 5, 10, 15, 20, 30],dtype=np.int32)
y = 100*np.array([0.191780, 0.247973, 0.392447, 0.456535, 0.536555, 0.645337])
e = 100*np.array([0.176787, 0.199224, 0.225419, 0.218645, 0.221468, 0.223282])

y2 = 100*np.array([0, 0.076156, 0.098249, 0.114167, 0.151805, 0.156505])
e2 = 100*np.array([0, 0.115459, 0.131384, 0.145297, 0.152405, 0.182161])

plt.errorbar(x, y, e, linestyle='None', marker='^')
lines = plt.errorbar(x, y2, e2, linestyle='None', marker='*')
plt.setp(lines, color='r', linewidth=2.0)

plt.ylabel('Porcentagem (%)')
plt.xlabel('# de exemplos de treino por classe')

plt.savefig('results.png', bbox_inches='tight')

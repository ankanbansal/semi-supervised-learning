import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

r_t = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mel_accuracy = [36.71, 39.91, 41.27, 41.5, 42.26, 45.22, 44.69, 41.72]
baseline_accuracy = [42.04]*8

plt.plot(r_t, mel_accuracy, '-rs', r_t, baseline_accuracy, '-b')
for a,b in zip(r_t, mel_accuracy): 
    plt.text(a, b, str(b))
plt.grid()
plt.legend([r'$ L_C + \alpha L_M $', r'$L_C$'], loc=2)
plt.xlabel('R/T')
plt.ylabel(r'Accuracy ($\%$)')
plt.title(r'Accuracy for baseline and model trained with $L_C + \alpha L_M$ for different $R/T$')
plt.savefig('accuracies.pdf')
plt.close()


num_unsup = [130000, 200000, 250000, 400000]
accuracy = [41.8, 42.26, 42.69, 45.68]
plt.plot(num_unsup, accuracy, '-rs')
for a,b in zip(num_unsup, accuracy): 
    plt.text(a, b, str(b))
plt.grid()
plt.xlabel(r'$|\mathcal{U}|$')
plt.ylabel(r'Accuracy ($\%$)')
plt.title('Accuracy for different number of unsupervised images')
plt.savefig('accuracies_unsup.pdf')
plt.close()


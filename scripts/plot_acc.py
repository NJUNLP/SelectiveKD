import pandas as pd
import math
import matplotlib.pyplot as plt

def plot_acc():
    with open("acc_rec") as f:
        acc_rec = f.readlines()
    
    acc_logs = []
    for rec in acc_rec:
        acc_logs.append(float(rec.strip()))
    print(len(acc_logs))
    
    rec = {i: 0 for i in range(10)}
    for acc in acc_logs:
        acc_class = int((acc) * 100 // 10)
        acc_class = 9 if acc_class == 10 else acc_class
        rec[acc_class] += 1
    total = len(acc_logs)

    x = ["[{:.1f}, {:.1f})".format(0.1 * i, 0.1 * (i + 1)) for i in range(10)]
    x[-1] = "[0.9, 1.0]"
    y = [rec[i] / total for i in range(10)]

    for i in range(1, len(y)):
        y[i] += y[i-1]

    plt.figure()
    plt.tight_layout()
    plt.bar(x, y)
    plt.xticks(rotation=30)
    plt.ylabel("Ratio")
    plt.xlabel("Prediction Accuracy")
    plt.savefig("acc.png", dpi=500)
    plt.close()


plot_acc()
# Create four separate plots with larger axis fonts
import matplotlib.pyplot as plt


sigma_labels = [0, 0.005, 0.01, 0.015, 0.02,0.025]
x = range(len(sigma_labels))

# Data

pacs_res18   = [80.56, 81.33, 81.86, 81.81, 81.69,81.47]
pacs_res50   = [85.34, 85.93, 86.09, 86.11, 86.08,86.01]
office_res18 = [63.41, 63.45, 63.53, 63.41, 63.25,63.18]
office_res50 = [69.96, 70.02, 70.10, 69.71, 69.54,69.37]

def plot_single(y, label, filename, marker,color):
    plt.figure(figsize=(5, 4.5))
    plt.plot(x, y, marker=marker, color=color, linewidth=2, label=label)
    plt.xticks(x, sigma_labels, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(r"$\sigma$", fontsize=20)
    plt.ylabel("Accuracy (%)", fontsize=20)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(fontsize=14, loc="best")
    plt.tight_layout()
    plt.savefig(f"{filename}", dpi=300, bbox_inches="tight")
    plt.show()

# Four figures
plot_single(pacs_res18,   "PACS—ResNet-18",      "pacs_res18_sigma.png",   "o", "black")
plot_single(pacs_res50,   "PACS—ResNet-50",      "pacs_res50_sigma.png",   "s", "blue")
plot_single(office_res18, "Office-Home—ResNet-18","officehome_res18_sigma.png", "^", "red")
plot_single(office_res50, "Office-Home—ResNet-50","officehome_res50_sigma.png", "D", "green")
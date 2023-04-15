import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create dataset
data = {'Environment': ['City', 'City', 'Forest', 'Forest', 'Suburban', 'Suburban'],
        'Algorithm': ['New', 'Traditional', 'New', 'Traditional', 'New', 'Traditional'],
        'APS': [0.50, 0.68, 0.75, 0.81, 0.39, 0.4],
        'ANSES': [30.79, 60, 47.02, 72.17, 22.27, 30.04]}

df = pd.DataFrame(data)

# Set up the seaborn style
sns.set(style="whitegrid")

# Create bar plots for APS and ANSES
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot APS scores
sns.barplot(x="Environment", y="APS", hue="Algorithm", data=df, ax=axs[0])
axs[0].set_title("APS Scores")
axs[0].set_xlabel("Environment")
axs[0].set_ylabel("APS")

# Plot ANSES scores
sns.barplot(x="Environment", y="ANSES", hue="Algorithm", data=df, ax=axs[1])
axs[1].set_title("ANSES Scores")
axs[1].set_xlabel("Environment")
axs[1].set_ylabel("ANSES")

# Add a legend and show the plots
plt.legend(title="Algorithm", loc="upper left")
plt.tight_layout()
plt.show()
# -----------------------------------
# IRIS DATASET VISUALIZATION
# USING PANDAS, MATPLOTLIB & SEABORN
# -----------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


# 1. INITIALIZATION & DIRECTORY SETUP

# Define and create output folders cleanly
base_output_folder = "output"
matplotlib_folder = os.path.join(base_output_folder, "matplotlib")
seaborn_folder = os.path.join(base_output_folder, "seaborn")

os.makedirs(matplotlib_folder, exist_ok=True)
os.makedirs(seaborn_folder, exist_ok=True)


# 2. DATA LOADING & PREPROCESSING

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df["species"] = iris.target
df["species"] = df["species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

# Display Dataset Information
print("=" * 50)
print("IRIS DATASET SUMMARY")
print("=" * 50)
print("\nFirst Five Rows:\n", df.head())
print("\nDataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe())
print("-" * 50)



# 3. MATPLOTLIB VISUALIZATIONS

print("\nGenerating Matplotlib Visualizations...")

# 1. Line Plot
plt.figure(figsize=(8, 5))
plt.plot(df["sepal length (cm)"], label="Sepal Length")
plt.title("Line Plot - Sepal Length")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length")
plt.legend()
plt.grid(True)
plt.savefig(f"{matplotlib_folder}/line_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 2. Scatter Plot
plt.figure(figsize=(8, 5))
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"], color="red")
plt.title("Scatter Plot")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.grid(True)
plt.savefig(f"{matplotlib_folder}/scatter_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 3. Bar Chart
species_count = df["species"].value_counts()
plt.figure(figsize=(7, 5))
plt.bar(species_count.index, species_count.values, color=["blue", "green", "orange"])
plt.title("Species Count")
plt.xlabel("Species")
plt.ylabel("Count")
plt.savefig(f"{matplotlib_folder}/bar_chart.png", dpi=300, bbox_inches="tight")
plt.show()

# 4. Horizontal Bar Chart
plt.figure(figsize=(7, 5))
plt.barh(species_count.index, species_count.values, color=["purple", "cyan", "gold"])
plt.title("Species Count")
plt.xlabel("Count")
plt.ylabel("Species")
plt.savefig(f"{matplotlib_folder}/horizontal_bar_chart.png", dpi=300, bbox_inches="tight")
plt.show()

# 5. Histogram
plt.figure(figsize=(8, 5))
plt.hist(df["sepal length (cm)"], bins=10, edgecolor="black")
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.savefig(f"{matplotlib_folder}/histogram.png", dpi=300, bbox_inches="tight")
plt.show()

# 6. Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(species_count.values, labels=species_count.index, autopct="%1.1f%%", startangle=90)
plt.title("Species Distribution")
plt.savefig(f"{matplotlib_folder}/pie_chart.png", dpi=300, bbox_inches="tight")
plt.show()

# 7. Box Plot
plt.figure(figsize=(8, 5))
plt.boxplot([
    df["sepal length (cm)"],
    df["sepal width (cm)"],
    df["petal length (cm)"],
    df["petal width (cm)"]
], tick_labels=[  # Updated from 'labels' to 'tick_labels'
    "Sepal Length", "Sepal Width", "Petal Length", "Petal Width"
])
plt.title("Box Plot")
plt.savefig(f"{matplotlib_folder}/box_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 8. Area Plot
plt.figure(figsize=(8, 5))
x = range(len(df))
plt.fill_between(x, df["petal width (cm)"], color="skyblue")
plt.title("Area Plot")
plt.xlabel("Sample")
plt.ylabel("Petal Width")
plt.savefig(f"{matplotlib_folder}/area_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 9. Stem Plot
plt.figure(figsize=(8, 5))
plt.stem(df.index[:20], df["petal width (cm)"][:20])
plt.title("Stem Plot")
plt.xlabel("Sample")
plt.ylabel("Petal Width")
plt.savefig(f"{matplotlib_folder}/stem_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 10. Step Plot
plt.figure(figsize=(8, 5))
plt.step(df.index[:30], df["sepal width (cm)"][:30], where="mid")
plt.title("Step Plot")
plt.xlabel("Sample")
plt.ylabel("Sepal Width")
plt.grid(True)
plt.savefig(f"{matplotlib_folder}/step_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 11. Multiple Line Plot
plt.figure(figsize=(10, 5))
plt.plot(df["sepal length (cm)"], label="Sepal Length")
plt.plot(df["petal length (cm)"], label="Petal Length")
plt.legend()
plt.title("Multiple Line Plot")
plt.xlabel("Sample")
plt.ylabel("Length")
plt.grid(True)
plt.savefig(f"{matplotlib_folder}/multiple_line_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 12. Subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].hist(df["sepal length (cm)"])
axes[0, 0].set_title("Sepal Length")
axes[0, 1].hist(df["sepal width (cm)"])
axes[0, 1].set_title("Sepal Width")
axes[1, 0].hist(df["petal length (cm)"])
axes[1, 0].set_title("Petal Length")
axes[1, 1].hist(df["petal width (cm)"])
axes[1, 1].set_title("Petal Width")
plt.tight_layout()
plt.savefig(f"{matplotlib_folder}/subplots.png", dpi=300, bbox_inches="tight")
plt.show()

print("All Matplotlib visualizations generated successfully.")



# 4. SEABORN VISUALIZATIONS

print("\nGenerating Seaborn Visualizations...")
sns.set_style("whitegrid")

# 1. Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species")
plt.title("Scatter Plot")
plt.savefig(f"{seaborn_folder}/scatter_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 2. Line Plot
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x=df.index, y="sepal length (cm)")
plt.title("Line Plot")
plt.savefig(f"{seaborn_folder}/line_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 3. Bar Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="species", y="petal length (cm)")
plt.title("Bar Plot")
plt.savefig(f"{seaborn_folder}/bar_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 4. Count Plot
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="species")
plt.title("Count Plot")
plt.savefig(f"{seaborn_folder}/count_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 5. Histogram
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="sepal length (cm)", bins=10, kde=True)
plt.title("Histogram")
plt.savefig(f"{seaborn_folder}/histogram.png", dpi=300, bbox_inches="tight")
plt.show()

# 6. Box Plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="species", y="petal width (cm)")
plt.title("Box Plot")
plt.savefig(f"{seaborn_folder}/box_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 7. Violin Plot
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="species", y="petal length (cm)")
plt.title("Violin Plot")
plt.savefig(f"{seaborn_folder}/violin_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 8. Strip Plot
plt.figure(figsize=(8, 5))
sns.stripplot(data=df, x="species", y="petal width (cm)")
plt.title("Strip Plot")
plt.savefig(f"{seaborn_folder}/strip_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 9. Swarm Plot
plt.figure(figsize=(8, 5))
sns.swarmplot(data=df, x="species", y="petal width (cm)", size=4) # Added size=4
plt.title("Swarm Plot")
plt.savefig(f"{seaborn_folder}/swarm_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 10. KDE Plot
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x="sepal length (cm)", fill=True)
plt.title("KDE Plot")
plt.savefig(f"{seaborn_folder}/kde_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 11. Pair Plot
pair_plot = sns.pairplot(df, hue="species")
pair_plot.savefig(f"{seaborn_folder}/pairplot.png", dpi=300, bbox_inches="tight")
plt.show()

# 12. Heatmap
plt.figure(figsize=(8, 6))
correlation = df.drop("species", axis=1).corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(f"{seaborn_folder}/heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# 13. Regression Plot
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x="sepal length (cm)", y="petal length (cm)")
plt.title("Regression Plot")
plt.savefig(f"{seaborn_folder}/regression_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 14. Joint Plot
joint_plot = sns.jointplot(data=df, x="sepal length (cm)", y="petal length (cm)", kind="scatter")
joint_plot.savefig(f"{seaborn_folder}/joint_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 15. Rug Plot
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x="petal width (cm)", fill=True)
sns.rugplot(data=df, x="petal width (cm)")
plt.title("Rug Plot")
plt.savefig(f"{seaborn_folder}/rug_plot.png", dpi=300, bbox_inches="tight")
plt.show()

print("All Seaborn visualizations generated successfully.")
print("=" * 50)
print(f"Process complete! Images are located under: \n- {matplotlib_folder}\n- {seaborn_folder}")
print("=" * 50)
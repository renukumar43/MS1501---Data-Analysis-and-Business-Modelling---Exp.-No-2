# Exp No.2 : Exploratory Data Analysis (EDA)

## Aim
To perform Exploratory Data Analysis (EDA) on the heart disease dataset to understand the data distribution, relationships, and key patterns useful for business insights and model building.

## Dataset
The dataset used for this experiment is `heartdisease.csv`. It contains medical and demographic information such as age, sex, cholesterol levels (`totChol`), `BMI`, blood pressure, and target values (`TenYearCHD`) indicating heart disease presence.

## Algorithm
1. Import required Python libraries (`numpy`, `pandas`, `seaborn`, `matplotlib`).
2. Load the dataset into a Pandas DataFrame.
3. Display the dataset structure, shape, and descriptive statistics.
4. Check for missing values and duplicate rows.
5. Examine correlations among numeric features using a correlation matrix and heatmap.
6. Visualize univariate distributions using histograms, boxplots, and countplots.
7. Analyze bivariate and multivariate relationships using scatter plots and pairplots.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Basic Exploration** | `df.info()` | Inspect dataset structure and data types |
| | `df.describe()` | Generate descriptive statistics |
| | `df.isnull().sum()` | Detect missing values in each column |
| | `df.duplicated().sum()` | Identify duplicate rows |
| | `df.corr()` | Find correlations between numeric variables |
| **Univariate Analysis** | `sns.histplot()` | Plot histogram (distribution of single variable) |
| | `sns.boxplot()` | Identify outliers using boxplot |
| | `sns.countplot()` | Count occurrences of categorical variables |
| **Bivariate Analysis** | `sns.scatterplot()` | Display relationship between two variables |
| | `sns.heatmap()` | Show correlation matrix visually |
| **Multivariate Analysis** | `sns.pairplot()` | Show pairwise relationships across variables |
| **Visualization Controls** | `plt.figure()`, `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.show()` | Configure and display plotted charts |

## Program & Output

# 1. Import required libraries
import numpy as np  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  

sns.set_theme(style="whitegrid")  
plt.rcParams["figure.figsize"] = (8, 5)  

print("Libraries imported successfully.")  

# 2. Load the dataset
file_path = "heartdisease (1).csv"  # Change to "heartdisease.csv" if that is your file name
df = pd.read_csv(file_path)  

print("Dataset loaded successfully.")  
display(df.head())  

# 3. Display dataset structure and shape
print("Dataset shape (rows, columns):", df.shape)  
print("\nDataset information:")  
df.info()  

print("\nColumn names:")  
print(df.columns.tolist())  

# 4. Descriptive statistics
print("Descriptive statistics:")  
display(df.describe(include="all"))  

# 5. Check missing values and duplicate rows
print("Missing values in each column:")  
display(df.isnull().sum())  

print("Total missing values:", df.isnull().sum().sum())  
print("Duplicate rows:", df.duplicated().sum())  

# 6. Correlation matrix and heatmap
numeric_df = df.select_dtypes(include=np.number)  
correlation_matrix = numeric_df.corr()  

display(correlation_matrix)  

plt.figure(figsize=(13, 9))  
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)  
plt.title("Correlation Heatmap of Numeric Features")  
plt.tight_layout()  
plt.show()  

# 7. Univariate analysis: histograms
# Plot histograms for numeric columns (up to 12 columns to keep the output readable).
hist_columns = numeric_df.columns[:12].tolist()

for column in hist_columns:
    plt.figure(figsize=(7, 4))  
    sns.histplot(data=df, x=column, kde=True, bins=20)  
    plt.title(f"Distribution of {column}")  
    plt.xlabel(column)  
    plt.ylabel("Frequency")  
    plt.tight_layout()  
    plt.show()  


# 8. Univariate analysis: boxplots
# Boxplots help identify possible outliers in numeric variables.
for column in hist_columns:
    plt.figure(figsize=(7, 3))  
    sns.boxplot(data=df, x=column)  
    plt.title(f"Boxplot of {column}")  
    plt.xlabel(column)  
    plt.tight_layout()  
    plt.show()  

    # 9. Countplot for categorical columns and target
categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Include binary target/indicator columns where available.
for candidate in ["Gender", "TenYearCHD"]:
    if candidate in df.columns and candidate not in categorical_columns:
        categorical_columns.append(candidate)


for column in categorical_columns:
    plt.figure(figsize=(7, 4))  
    sns.countplot(data=df, x=column)  
    plt.title(f"Count of {column}")  
    plt.xlabel(column)  
    plt.ylabel("Count")  
    plt.tight_layout()  
    plt.show()  


    # 10. Bivariate analysis: scatter plots
# Use relevant columns if they exist in the dataset.
scatter_pairs = [
    ("Age", "Totalcholestrol"),
    ("Age", "BMI"),
    ("SysBP", "DiaBP"),
    ("Age", "Glucose")
]

available_pairs = [(x, y) for x, y in scatter_pairs if x in df.columns and y in df.columns]

if available_pairs:
    for x_col, y_col in available_pairs:
        plt.figure(figsize=(7, 5))  
        if "TenYearCHD" in df.columns:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue="TenYearCHD")  
            plt.title(f"{y_col} vs {x_col} by TenYearCHD")  
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col)  
            plt.title(f"{y_col} vs {x_col}")  
        plt.tight_layout()  
        plt.show()  
else:
    print("No matching column pairs were found for scatter plots.")  

    # 11. Multivariate analysis: pairplot
# Select a small set of useful numeric columns to keep the pairplot readable.
preferred_pairplot_cols = [
    col for col in ["Age", "Totalcholestrol", "BMI", "SysBP", "DiaBP", "TenYearCHD"]
    if col in df.columns
]

if len(preferred_pairplot_cols) >= 2:  
    pairplot_data = df[preferred_pairplot_cols].dropna()  
    if "TenYearCHD" in pairplot_data.columns:
        sns.pairplot(pairplot_data, hue="TenYearCHD", diag_kind="hist")  
    else:
        sns.pairplot(pairplot_data, diag_kind="hist")  
    plt.show()  
else:
    print("Not enough selected numeric columns are available for a pairplot.")  

    # 12. Simple target-wise summary (if target column exists)
if "TenYearCHD" in df.columns:  
    print("Average numeric values grouped by TenYearCHD:")  
    display(df.groupby("TenYearCHD")[numeric_df.columns].mean())  
else:
    print("TenYearCHD target column is not available in this dataset.")  

## Result

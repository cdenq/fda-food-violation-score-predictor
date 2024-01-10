#----------------------------------------------------
# Imports
#----------------------------------------------------
from modules import imports
from modules import helper

#----------------------------------------------------
# Helper Functions for EDA
#----------------------------------------------------
def check_values_perc(df: imports.pd.DataFrame, n: int=10) -> None:
    """ 
    Prints out dataframe of the value_counts as a percentage and the list of unique values.
    Used for briefly checking if values are in the right formats.
    n is set low to prevent VSCode grayscreen bug, which occurs when terminal memory caps 

    df -> Pandas.DataFrame
        Given dataframe to check

    n -> int
        The number of results to print

    Returns -> None
        prints the results to the terminal

    Example
        check_values_perc(df, n=10)
    """
    for col in df.columns:
        print(df[col].value_counts(normalize=True))
        print("")
        print(list(df[col].unique())[:n])
        print("---------")
    
    return None

def check_missing_nan(df: imports.pd.DataFrame) -> imports.pd.DataFrame:
    """ 
    Generates a dataframe showing the missing and Nan values in it.

    df -> Pandas.DataFrame
        Given dataframe to check

    Returns -> Pandas.DataFrame
        Dataframe with the results

    Example
        check_missing_nan(df)
    """
    # Checking for missing or NANs
    missing_cols, missing_rows = (
        (df.isnull().sum(x) | df.eq('').sum(x))
        .loc[lambda x: x > 0].index
        for x in (0, 1)
    )
    report = df.loc[missing_rows, missing_cols]
    if report.empty:
        print("No missing or NaN values!")

    return report

def graph_corr_matrix(df: imports.pd.DataFrame) -> None:
    """
    Generates and saves a correlation matrix of a given Pandas Dataframe

    df -> Pandas.DataFrame
        Given dataframe with only numerical values

    Returns -> None
        Plots and shows the correlation matrix
        Saves the correlation matrix to EDA folder

    Example
        graph_corr_matrix(df)
    """
    # Creating corr
    correlation_matrix = df.corr()

    # Pre Graph Formatting
    imports.plt.figure(figsize=imports.DEFAULT_BIG_FIG_SIZE)
    imports.plt.title(f"Correlation Matrix Heatmap")

    # Graphing
    mask = imports.np.triu(imports.np.ones_like(correlation_matrix, dtype=bool))
    imports.sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='Blues', fmt=".2f", cbar_kws={'shrink': .5})

    # Post Graph Formatting & Saving
    filepath = f"{imports.DEV_PATH_TO_EDA}/correlation_matrix.png"
    imports.plt.savefig(filepath)
    imports.plt.show()

    return None

def graph_description(values: imports.pd.Series, value_label: str) -> imports.Image:
    """
    Generates and saves the descriptive statistics of a given Pandas Series

    values -> Pandas.Series
        Given list of iterable values

    value_label -> str
        Name of the what the values represents

    Returns -> IPython.core.display.Image
        Saves the table to EDA folder
        Returns the IPython.core.display.Image to terminal

    Example
        graph_description(df["numeric_values"], "name_of_values")
    """
    # Generate descriptive statistics
    table_df = imports.pd.DataFrame(values.describe())
    table_df.rename(columns={table_df.columns[0]: value_label}, inplace=True)

    # Export
    filepath = f"{imports.DEV_PATH_TO_EDA}/descr_stat_{value_label.replace(" ", "_").lower()}.png"
    imports.dfi.export(table_df, filepath)

    return imports.Image(filepath)

def graph_histogram(values: imports.pd.Series, value_label: str) -> None:
    """
    Generates and saves a histogram of a given Pandas Series

    values -> Pandas.Series
        Given list of iterable values

    value_label -> str
        Name of the what the values represents

    Returns -> None
        Plots and shows the histogram
        Saves the histogram to EDA folder

    Example
        graph_histogram(df["values"], "name_of_values")
    """
    # Pre Graph Formatting
    imports.plt.figure(figsize=imports.DEFAULT_LONG_FIG_SIZE)
    imports.plt.title(f"Distribution of {value_label}")
    imports.plt.xlabel(f"{value_label}")
    imports.plt.ylabel("Frequency")
    imports.plt.grid("True", alpha=imports.DEFAULT_GRID_ALPHA)

    min_score = min(values)
    max_score = max(values)
    num_scores = len(values)
    num_bins = helper.calc_num_bins(num_scores)
    bin_edges = imports.np.linspace(min_score, max_score, num_bins)

    # Graphing
    imports.plt.hist(values, alpha=imports.DEFAULT_GRAPH_ALPHA, bins=bin_edges)

    # Post Graph Formatting & Saving
    imports.plt.tight_layout()
    filepath = f"{imports.DEV_PATH_TO_EDA}/histogram_{value_label.replace(" ", "_").lower()}.png"
    imports.plt.savefig(filepath)
    imports.plt.show()

    return None

def graph_boxplot(values: imports.pd.Series, value_label: str) -> None:
    """
    Generates and saves a boxplot of a given Pandas Series

    values -> Pandas.Series
        Given list of iterable values

    value_label -> str
        Name of the what the values represents

    Returns -> None
        Plots and shows the boxplot
        Saves the boxplot to EDA folder

    Example
        graph_boxplot(df["values"], "name_of_values")
    """
    # Pre Graph Formatting
    imports.plt.figure(figsize=imports.DEFAULT_TALL_FIG_SIZE)
    imports.plt.title(f"Outliers in {value_label}")
    imports.plt.ylabel(f"{value_label} Values")
    imports.plt.grid("True", alpha=imports.DEFAULT_GRID_ALPHA)

    labels = value_label
    outliers_format = {'marker': 'D', 'markerfacecolor': 'red', 'markersize': imports.DEFAULT_MARKER_SIZE}

    # Graphing
    imports.plt.boxplot(values, notch=True, labels=[labels], flierprops=outliers_format)

    # Post Graph Formatting & Saving
    imports.plt.tight_layout()
    filepath = f"{imports.DEV_PATH_TO_EDA}/boxplot_{value_label.replace(" ", "_").lower()}.png"
    imports.plt.savefig(filepath)
    imports.plt.show()

    return None

#----------------------------------------------------
# Main
#----------------------------------------------------
def main():
    return None

#----------------------------------------------------
# Entry
#----------------------------------------------------
if __name__ == "__main__":
    main()
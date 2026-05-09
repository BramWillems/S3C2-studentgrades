import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns   
import inspect

def analyze_missing_data(df, plot=True):
    """
    Analyze and visualize missing data in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze.
    plot : bool, optional (default=True)
        Whether to show plots for missing values.
    
    Returns:
    --------
    dict
        Summary statistics about missing data.
    """
    
    # Try to infer the variable name of the DataFrame
    frame = inspect.currentframe().f_back
    df_name = None
    for var_name, var_val in frame.f_locals.items():
        if var_val is df:
            df_name = var_name
            break
    if df_name is None:
        df_name = "DataFrame"

    # Total missing data points
    total_missing = df.isnull().sum().sum()
    
    # Total data points
    total_data = df.size
    
    # Percentage of missing data overall
    percent_missing_overall = (total_missing / total_data) * 100
    
    # Percentage of missing data per column
    missing_per_column = (df.isnull().sum() / df.shape[0]) * 100
    missing_per_column = missing_per_column.sort_values(ascending=False)
    
    # Print summary
    print(f"--- Missing Data Report for {df_name} ---")
    print(f"Total Data Points: {total_data}")
    print(f"Total Missing Data Points: {total_missing}")
    print(f"Overall Missing Data: {percent_missing_overall:.2f}%\n")
    print("Missing Data Per Column (%):")
    
    if plot:
        # Plot missing per column
        if missing_per_column.sum() > 0:
            plt.figure(figsize=(12,6))
            sns.barplot(x=missing_per_column.index, y=missing_per_column.values)
            plt.xticks(rotation=90)
            plt.ylabel("Missing Data (%)")
            plt.title(f"Percentage of Missing Data per Column - {df_name}")
            plt.show()
        
        # Heatmap of missing values
        plt.figure(figsize=(14, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title(f"Missing Values Heatmap - {df_name}")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()
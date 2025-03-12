# src/analysis/univariate.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def enhanced_telecom_analysis(df, dataset_name, figsize=(12, 8), save_results=True):
    """
    Perform enhanced univariate analysis specifically for telecom datasets
    
    Parameters:
    -----------
    df : pandas DataFrame
        The telecom dataset to analyze
    dataset_name : str
        Name of the dataset for titles
    figsize : tuple
        Base figure size
    save_results : bool
        Whether to save results to files
    """
    # Create output directories if saving results
    if save_results:
        # Get project root directory (assuming this function is in src/analysis/univariate.py)
        current_file_path = os.path.abspath(__file__)
        src_dir = os.path.dirname(os.path.dirname(current_file_path))
        project_root = os.path.dirname(src_dir)
        
        # Create paths for saving results
        dataset_dir_name = dataset_name.lower().replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define directories
        reports_dir = os.path.join(project_root, 'reports', 'univariate_analysis')
        base_dir = os.path.join(reports_dir, dataset_dir_name, timestamp)
        plots_dir = os.path.join(base_dir, 'plots')
        stats_dir = os.path.join(base_dir, 'statistics')
        
        # Create directories
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)
        
        # Redirect stdout to capture print statements
        original_stdout = sys.stdout
        log_file = open(os.path.join(base_dir, "analysis_log.txt"), 'w')
        sys.stdout = log_file
    
    print(f"\n{'='*80}\n{dataset_name.upper()} ANALYSIS\n{'='*80}")
    print(f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Dataset Overview
    print(f"\nDataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Save dataset info if requested
    if save_results:
        # Capture info to file
        buffer = io.StringIO()
        df.info(buf=buffer)
        with open(os.path.join(stats_dir, "dataset_info.txt"), 'w') as f:
            f.write(buffer.getvalue())
        
        # Save dataset head
        df.head(20).to_csv(os.path.join(stats_dir, "dataset_head.csv"))
        
        # Save dataset description
        with open(os.path.join(stats_dir, "dataset_description.txt"), 'w') as f:
            f.write(str(df.describe(include='all')))
    
    # Categorize variables
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 2. Missing Values Analysis
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_pct})
    missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
    
    if not missing_df.empty:
        print("\nMissing Values Analysis:")
        print(missing_df)
        
        # Save missing values data
        if save_results:
            missing_df.to_csv(os.path.join(stats_dir, "missing_values.csv"))
        
        # Visualize missing values
        plt.figure(figsize=figsize)
        sns.heatmap(df[missing_df.index].isnull(), cmap='YlOrRd', yticklabels=False)
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        
        if save_results:
            plt.savefig(os.path.join(plots_dir, "missing_values_heatmap.png"), dpi=300, bbox_inches='tight')
        
        plt.show()

    # 3. Numerical Variables Analysis
    def analyze_numerical_variables(columns, title):
        print(f"\n{title}:")
        stats_df = df[columns].describe()
        stats_df.loc['skewness'] = df[columns].skew()
        stats_df.loc['kurtosis'] = df[columns].kurtosis()
        print(stats_df.round(2))
        
        # Save statistics
        if save_results:
            stats_df.round(4).to_csv(os.path.join(stats_dir, f"{title.replace(' ', '_').lower()}.csv"))
        
        # Visualization
        for i in range(0, len(columns), 3):
            batch = columns[i:i+3]
            n_cols = len(batch)
            
            # Create figure with proper dimensions
            fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
            
            # Handle the case when there's only one column in the batch
            if n_cols == 1:
                axes = np.array(axes).reshape(2, 1)
            
            for idx, col in enumerate(batch):
                # Distribution plot
                sns.histplot(data=df, x=col, kde=True, ax=axes[0, idx])
                axes[0, idx].set_title(f'Distribution of {col}')
                
                # Add statistics
                stats_text = f'Mean: {df[col].mean():.2f}\n'
                stats_text += f'Median: {df[col].median():.2f}\n'
                stats_text += f'Std: {df[col].std():.2f}\n'
                stats_text += f'Skew: {df[col].skew():.2f}'
                
                axes[0, idx].text(0.95, 0.95, stats_text,
                                transform=axes[0, idx].transAxes,
                                verticalalignment='top',
                                horizontalalignment='right',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Boxplot
                sns.boxplot(data=df, y=col, ax=axes[1, idx])
                axes[1, idx].set_title(f'Boxplot of {col}')
                
                # Add outlier information
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
                outlier_pct = (len(outliers) / len(df) * 100)
                
                axes[1, idx].text(0.95, 0.95, f'Outliers: {outlier_pct:.1f}%',
                                transform=axes[1, idx].transAxes,
                                verticalalignment='top',
                                horizontalalignment='right',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            if save_results:
                batch_name = "_".join([col.replace(' ', '_').lower() for col in batch])
                plt.savefig(os.path.join(plots_dir, f"{title.replace(' ', '_').lower()}_{batch_name}.png"), 
                           dpi=300, bbox_inches='tight')
            
            plt.show()

    # Separate numerical variables into categories
    financial_cols = [col for col in numeric_cols if any(x in col.lower() 
                     for x in ['charge', 'revenue', 'cltv', 'refund', 'total'])]
    usage_cols = [col for col in numeric_cols if any(x in col.lower() 
                 for x in ['download', 'data', 'tenure', 'score'])]
    demographic_cols = [col for col in numeric_cols if any(x in col.lower() 
                      for x in ['age', 'dependent', 'population', 'citizen'])]
    
    # Other numerical columns that don't fit into the above categories
    other_numeric_cols = [col for col in numeric_cols if col not in financial_cols + usage_cols + demographic_cols]
    
    # Save variable categorization
    if save_results:
        with open(os.path.join(stats_dir, "variable_categories.txt"), 'w') as f:
            f.write("Financial Variables:\n")
            f.write(", ".join(financial_cols) + "\n\n")
            f.write("Usage Variables:\n")
            f.write(", ".join(usage_cols) + "\n\n")
            f.write("Demographic Variables:\n")
            f.write(", ".join(demographic_cols) + "\n\n")
            f.write("Other Numerical Variables:\n")
            f.write(", ".join(other_numeric_cols) + "\n\n")
            f.write("Categorical Variables:\n")
            f.write(", ".join(categorical_cols))
    
    # Analyze each category
    if financial_cols:
        analyze_numerical_variables(financial_cols, "Financial Metrics Analysis")
    if usage_cols:
        analyze_numerical_variables(usage_cols, "Usage Metrics Analysis")
    if demographic_cols:
        analyze_numerical_variables(demographic_cols, "Demographic Metrics Analysis")
    if other_numeric_cols:
        analyze_numerical_variables(other_numeric_cols, "Other Numerical Metrics Analysis")

    # 4. Categorical Variables Analysis
    if len(categorical_cols) > 0:
        print("\nCategorical Variables Analysis:")
        
        # Group categorical variables by number of unique values
        binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
        multi_cols = [col for col in categorical_cols if df[col].nunique() > 2 and df[col].nunique() <= 10]
        high_card_cols = [col for col in categorical_cols if df[col].nunique() > 10]
        
        # Save categorical variable grouping
        if save_results:
            with open(os.path.join(stats_dir, "categorical_variables.txt"), 'w') as f:
                f.write("Binary Variables:\n")
                f.write(", ".join(binary_cols) + "\n\n")
                f.write("Multi-Category Variables:\n")
                f.write(", ".join(multi_cols) + "\n\n")
                f.write("High Cardinality Variables:\n")
                f.write(", ".join(high_card_cols))
        
        # Analyze binary categorical variables
        if binary_cols:
            print("\nBinary Categorical Variables:")
            for col in binary_cols:
                value_counts = df[col].value_counts()
                value_percentages = (value_counts / len(df) * 100).round(2)
                
                print(f"\n{col}:")
                summary_df = pd.DataFrame({
                    'Count': value_counts,
                    'Percentage': value_percentages
                })
                print(summary_df)
                
                # Save statistics
                if save_results:
                    summary_df.to_csv(os.path.join(stats_dir, f"categorical_{col.replace(' ', '_').lower()}.csv"))
                
                # Visualization
                plt.figure(figsize=(8, 6))
                ax = sns.countplot(data=df, x=col)
                plt.title(f'Distribution of {col}')
                
                # Add percentage labels
                for i, v in enumerate(value_counts.values):
                    ax.text(i, v, f'{value_percentages.iloc[i]:.1f}%', 
                           ha='center', va='bottom')
                
                plt.tight_layout()
                
                if save_results:
                    plt.savefig(os.path.join(plots_dir, f"categorical_{col.replace(' ', '_').lower()}.png"), 
                               dpi=300, bbox_inches='tight')
                
                plt.show()
        
        # Analyze multi-category variables
        if multi_cols:
            print("\nMulti-Category Variables:")
            for col in multi_cols:
                value_counts = df[col].value_counts()
                value_percentages = (value_counts / len(df) * 100).round(2)
                
                print(f"\n{col}:")
                summary_df = pd.DataFrame({
                    'Count': value_counts,
                    'Percentage': value_percentages
                })
                print(summary_df)
                
                # Save statistics
                if save_results:
                    summary_df.to_csv(os.path.join(stats_dir, f"categorical_{col.replace(' ', '_').lower()}.csv"))
                
                # Visualization
                plt.figure(figsize=(10, 6))
                ax = sns.countplot(data=df, x=col)
                plt.title(f'Distribution of {col}')
                plt.xticks(rotation=45, ha='right')
                
                # Add percentage labels
                for i, v in enumerate(value_counts.values):
                    ax.text(i, v, f'{value_percentages.iloc[i]:.1f}%', 
                           ha='center', va='bottom')
                
                plt.tight_layout()
                
                if save_results:
                    plt.savefig(os.path.join(plots_dir, f"categorical_{col.replace(' ', '_').lower()}.png"), 
                               dpi=300, bbox_inches='tight')
                
                plt.show()
        
        # Just mention high cardinality variables
        if high_card_cols:
            print("\nHigh Cardinality Categorical Variables:")
            for col in high_card_cols:
                print(f"{col}: {df[col].nunique()} unique values")
                
                # Save top values for high cardinality variables
                if save_results:
                    top_values = df[col].value_counts().head(20)
                    top_values.to_csv(os.path.join(stats_dir, f"top_values_{col.replace(' ', '_').lower()}.csv"))

    # 5. Churn Analysis (specific to telecom datasets)
    churn_col = None
    if 'Churn' in df.columns:
        churn_col = 'Churn'
    elif any(col.lower() == 'churn' for col in df.columns):
        churn_col = next(col for col in df.columns if col.lower() == 'churn')
    
    if churn_col:
        print("\nChurn Analysis:")
        
        # Convert to numeric if it's categorical
        if df[churn_col].dtype == 'object':
            if df[churn_col].nunique() == 2:
                # Try to convert Yes/No to 1/0
                if set(df[churn_col].unique()) == {'Yes', 'No'}:
                    churn_numeric = df[churn_col].map({'Yes': 1, 'No': 0})
                    churn_rate = (churn_numeric.mean() * 100).round(2)
                else:
                    # Just count the proportion of the first value
                    churn_rate = (df[churn_col] == df[churn_col].unique()[0]).mean() * 100
            else:
                print(f"Churn has {df[churn_col].nunique()} categories, showing distribution:")
                churn_rate = None
        else:
            churn_rate = (df[churn_col].mean() * 100).round(2)
        
        if churn_rate is not None:
            print(f"Overall Churn Rate: {churn_rate}%")
            
            # Save churn rate
            if save_results:
                with open(os.path.join(stats_dir, "churn_rate.txt"), 'w') as f:
                    f.write(f"Overall Churn Rate: {churn_rate}%")
        
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=churn_col)
        plt.title('Churn Distribution')
        
        # Add percentage labels
        total = len(df)
        for i, p in enumerate(df[churn_col].value_counts()):
            percentage = 100 * p / total
            plt.text(i, p, f'{percentage:.1f}%', ha='center', va='bottom')
            
        plt.tight_layout()
        
        if save_results:
            plt.savefig(os.path.join(plots_dir, "churn_distribution.png"), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    # Generate summary report
    if save_results:
        with open(os.path.join(base_dir, "analysis_summary.md"), 'w') as f:
            f.write(f"# {dataset_name} Analysis Summary\n\n")
            f.write(f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Dataset Overview\n")
            f.write(f"- Rows: {df.shape[0]}\n")
            f.write(f"- Columns: {df.shape[1]}\n")
            f.write(f"- Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB\n\n")
            
            f.write("## Variable Types\n")
            f.write(f"- Numerical Variables: {len(numeric_cols)}\n")
            f.write(f"- Categorical Variables: {len(categorical_cols)}\n\n")
            
            if not missing_df.empty:
                f.write("## Missing Values\n")
                f.write(f"- Variables with missing values: {len(missing_df)}\n")
                f.write(f"- Total missing values: {missing_df['Missing Values'].sum()}\n\n")
            
            if churn_col:
                f.write("## Churn Analysis\n")
                if churn_rate is not None:
                    f.write(f"- Overall Churn Rate: {churn_rate}%\n\n")
            
            f.write("## Key Insights\n")
            f.write("- Add your key insights here after reviewing the analysis\n\n")
            
            f.write("## Next Steps\n")
            f.write("- Add recommended next steps here\n")
        
        # Copy key figures to the main figures directory for reports
        figures_dir = os.path.join(project_root, 'reports', 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Copy churn distribution if it exists
        if churn_col and os.path.exists(os.path.join(plots_dir, "churn_distribution.png")):
            import shutil
            shutil.copy2(
                os.path.join(plots_dir, "churn_distribution.png"),
                os.path.join(figures_dir, f"{dataset_dir_name}_churn_distribution.png")
            )
        
        # Close the log file and restore stdout
        sys.stdout = original_stdout
        log_file.close()
        
        print(f"Analysis results saved to {base_dir}/")
        print(f"Key figures copied to {figures_dir}/")

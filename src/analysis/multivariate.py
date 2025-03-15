import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def perform_multivariate_analysis(df, dataset_name):
    # Get project root directory
    current_file_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(os.path.dirname(current_file_path))
    project_root = os.path.dirname(src_dir)
    
    # Create paths for saving results
    dataset_dir_name = dataset_name.lower().replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define directories
    reports_dir = os.path.join(project_root, 'reports', 'multivariate_analysis')
    base_dir = os.path.join(reports_dir, dataset_dir_name, timestamp)
    plots_dir = os.path.join(base_dir, 'plots')
    stats_dir = os.path.join(base_dir, 'statistics')
    
    # Create directories
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    # Open log file
    log_file = open(os.path.join(base_dir, "analysis_log.txt"), 'w')

    try:
        # Perform analysis
        log_file.write(f"Starting multivariate analysis for {dataset_name}\n")

        # Correlation analysis
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title(f"Correlation Matrix - {dataset_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "correlation_matrix.png"))
        plt.close()

        log_file.write("Correlation matrix created and saved.\n")

        # Pairplot for selected features
        selected_features = ['MonthlyCharges', 'TotalCharges', 'tenure']  # Adjust as needed
        if all(feature in df.columns for feature in selected_features + ['Churn']):
            sns.pairplot(df[selected_features + ['Churn']], hue='Churn')
            plt.suptitle(f"Pairplot of Selected Features - {dataset_name}", y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "pairplot.png"))
            plt.close()
            log_file.write("Pairplot created and saved.\n")
        else:
            log_file.write("Skipping pairplot due to missing columns.\n")

        # Chi-square test for categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        chi_square_results = []

        for col in categorical_cols:
            if col != 'Churn' and 'Churn' in df.columns:
                chi2, p, dof, expected = stats.chi2_contingency(pd.crosstab(df['Churn'], df[col]))
                chi_square_results.append({'Feature': col, 'Chi2': chi2, 'p-value': p})

        if chi_square_results:
            chi_square_df = pd.DataFrame(chi_square_results).sort_values('p-value')
            chi_square_df.to_csv(os.path.join(stats_dir, "chi_square_results.csv"), index=False)
            log_file.write("Chi-square test results saved.\n")
        else:
            log_file.write("No categorical variables for chi-square test.\n")

        # Outlier analysis for MonthlyCharges
        if 'MonthlyCharges' in df.columns:
            Q1 = df['MonthlyCharges'].quantile(0.25)
            Q3 = df['MonthlyCharges'].quantile(0.75)
            IQR = Q3 - Q1

            outliers = df[(df['MonthlyCharges'] < Q1 - 1.5 * IQR) | (df['MonthlyCharges'] > Q3 + 1.5 * IQR)]
            outlier_pct = len(outliers) / len(df) * 100
            outliers.to_csv(os.path.join(stats_dir, "monthly_charges_outliers.csv"), index=False)

            log_file.write(f"Outlier analysis completed. Outlier percentage: {outlier_pct:.2f}%.\n")
        else:
            log_file.write("Skipping outlier analysis due to missing 'MonthlyCharges' column.\n")

        # Document summary
        with open(os.path.join(base_dir, "multivariate_analysis_summary.md"), 'w') as f:
            f.write(f"# {dataset_name} Multivariate Analysis Summary\n\n")
            f.write(f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Key Results:\n")
            f.write("- Correlation matrix shows relationships between numerical features.\n")
            f.write("- Pairplot visualizes distributions and relationships among selected features.\n")
            f.write("- Chi-square tests highlight significant categorical predictors of churn.\n")
            if 'MonthlyCharges' in df.columns:
                f.write(f"- Outlier analysis revealed that {outlier_pct:.2f}% of records have unusual Monthly Charges.\n\n")
            f.write("## Recommendations for Next Steps:\n")
            f.write("- Further investigation into high-correlation pairs for feature reduction.\n")
            f.write("- Advanced feature engineering based on identified significant categorical features.\n")
            f.write("- Consider outlier handling strategies in predictive modeling.\n")

        log_file.write("Summary document created.\n")

    except Exception as e:
        log_file.write(f"An error occurred: {str(e)}\n")
    finally:
        log_file.close()

if __name__ == "__main__":
    # Get project root directory
    current_file_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(os.path.dirname(current_file_path))
    project_root = os.path.dirname(src_dir)
    
    # Define data paths
    data_dir = os.path.join(project_root, 'data', 'processed')
    telco_churn_path = os.path.join(data_dir, 'Telco_churn_with_reason_Dataset.csv')
    ibm_churn_path = os.path.join(data_dir, 'IBM_churn_pred_Dataset.csv')

    # Load datasets
    try:
        telco_churn_df = pd.read_csv(telco_churn_path, encoding='utf-8')
        perform_multivariate_analysis(telco_churn_df, "Telco Customer Churn Dataset")
    except Exception as e:
        print(f"Error processing Telco dataset: {str(e)}")

    try:
        ibm_churn_df = pd.read_csv(ibm_churn_path, encoding='utf-8')
        perform_multivariate_analysis(ibm_churn_df, "IBM Customer Churn Dataset")
    except Exception as e:
        print(f"Error processing IBM dataset: {str(e)}")
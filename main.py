import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from config import config


def prepare_output_directory(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def descriptive_analysis(data, variable, output_dir):
    desc_stats = data[variable].describe()
    print(f"Descriptive Statistics for {variable}:\n{desc_stats}")

    plt.figure(figsize=(10, 6))
    sns.histplot(data[variable], kde=True, color='blue', stat="density")
    plt.axvline(desc_stats['25%'], color='g', linestyle='--', label='25th Percentile (Q1)')
    plt.axvline(desc_stats['50%'], color='b', linestyle='-', label='50th Percentile (Q2, Median)')
    plt.axvline(desc_stats['75%'], color='g', linestyle='--', label='75th Percentile (Q3)')
    plt.legend()
    plt.title(f'Histogram with Bell Curve of {variable}')
    plt.xlabel(variable)
    plt.ylabel('Density')
    plt.savefig(os.path.join(output_dir, f'histogram_bell_curve_{variable}.png'))
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[variable], color='lightblue')
    plt.title(f'Boxplot of {variable}')
    plt.savefig(os.path.join(output_dir, f'boxplot_{variable}.png'))
    plt.show()


def plot_multiple_events_over_time(data, event_columns, output_dir):
    plt.figure(figsize=(10, 6))
    for event in event_columns:
        plt.plot(data['Year'], data[event], label=event)
    plt.title('Trend for Extreme Weather Events over Time')
    plt.xlabel('Year')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'all_events_trend.png'))
    plt.show()


def plot_correlation_matrix(data, selected_columns, output_dir):
    selected_df = data[selected_columns]
    correlation_matrix = selected_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, square=True, fmt='.2f',
                annot_kws={'size': 10})
    plt.title('Correlation Matrix of Selected Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.show()


def plot_time_series(time, series, title, xlabel, ylabel, window, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(time, series, label='Actual')

    if window:
        rolling_mean = series.rolling(window=window).mean()
        plt.plot(time, rolling_mean, color='red', label=f'Rolling Mean (window={window})')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    plot_filename = f'time_series_{title}.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.show()


def bivariate_analysis(data_path, output_dir, dependent_variable):
    output_file_name = 'bivariate_analysis_grand_total.txt'
    df = pd.read_csv(data_path)
    prepare_output_directory(output_dir)
    output_path = os.path.join(output_dir, output_file_name)

    with open(output_path, 'w') as file:
        for col in df.columns:
            if col not in IGNORE_COLUMNS:
                file.write(f"Bivariate Analysis for {dependent_variable} and {col}:\n")
                plt.figure(figsize=(8, 5))
                sns.scatterplot(data=df, x=dependent_variable, y=col)
                plt.title(f'Scatter Plot of {dependent_variable} and {col}')
                plot_filename = f'scatterplot_{dependent_variable}_{col}.png'
                plt.savefig(os.path.join(output_dir, plot_filename))
                plt.close()
                correlation = df[dependent_variable].corr(df[col])
                file.write(f"Correlation Coefficient: {correlation:.2f}\n\n")
                covariance = df[dependent_variable].cov(df[col])
                file.write(f"Covariance: {covariance:.2f}\n\n")
                file.write('-' * 50 + '\n\n')

                sns.regplot(x=dependent_variable, y=col, data=df)
                plt.title(f'Regression Plot of {dependent_variable} and {col}')
                plot_filename = f'regplot_{dependent_variable}_{col}.png'
                plt.savefig(os.path.join(output_dir, plot_filename))
                plt.close()


def exploratory_data_analysis(data_path, output_dir, main_variable, event_columns, independent_variables):
    df = pd.read_csv(data_path)
    prepare_output_directory(output_dir)

    descriptive_analysis(df, main_variable, output_dir)

    # Plot All Extreme Weather Events on One Graph
    plot_multiple_events_over_time(df, event_columns, output_dir)

    # Correlation Matrix
    selected_columns = event_columns + independent_variables
    plot_correlation_matrix(df, selected_columns, output_dir)

    # Time Series Plots
    for col in df.columns:
        if col not in ['Year', independent_variables]:
            plot_time_series(df['Year'], df[col], f'Time-Series of {col}', 'Year', col, window=5,
                             output_dir=output_dir)


def print_multicollinearity(X, threshold=5.0):
    vif = pd.DataFrame()
    vif["Variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    high_vif = vif[vif['VIF'] > threshold]
    if not high_vif.empty:
        print("Warning: Potential multicollinearity detected:")
        print(high_vif)


def fit_regression_model(X, y):
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit()


def analyze_model_results(model, file, alpha=0.05):
    results_summary = model.summary2().tables[1]

    file.write("Automated Analysis of the Regression Results\n")
    file.write("-" * 45 + "\n")

    rsquared = model.rsquared
    file.write(f"R-squared: {rsquared:.3f}\n")
    if rsquared >= 0.7:
        file.write("The model explains a significant amount of the response variable's variability.\n")
    elif rsquared >= 0.5:
        file.write("The model explains a moderate amount of the response variable's variability.\n")
    else:
        file.write("The model does not explain much of the response variable's variability.\n")

    significant_vars = results_summary[results_summary['P>|t|'] <= alpha].index.tolist()
    if 'const' in significant_vars:
        significant_vars.remove('const')

    if significant_vars:
        file.write(f"\nSignificant predictors at alpha={alpha}: {', '.join(significant_vars)}.\n")
    else:
        file.write(f"\nNo significant predictors found at alpha={alpha}.\n")

    file.write("-" * 45 + "\n")


def regression_summary(data_path, dependent_variable, independent_variables, output_dir):
    dependent_variable_clean = dependent_variable.replace(" ", "_")
    output_file_name = f'regression_analysis_{dependent_variable_clean}.txt'

    df = pd.read_csv(data_path)
    prepare_output_directory(output_dir)
    output_path = os.path.join(output_dir, output_file_name)

    X = df[independent_variables]
    # print_multicollinearity(X)

    y = df[dependent_variable]
    model = fit_regression_model(X, y)

    with open(output_path, 'w') as file:
        file.write(model.summary().as_text() + "\n")
        analyze_model_results(model, file)


# Constants
IGNORE_COLUMNS = ['Year', 'Grand Total', 'Flood', 'Extreme weather', 'Drought', 'Extreme temperature', 'Wildfire']
INDEPENDENT_VARIABLES = ['Temperature Anomalies in Celsius',
                         'Average CO2 levels in the atmosphere worldwide(in ppm)',
                         'Annual CO2 emissions worldwide (in billion metric tons)',
                         'Adjusted sea level increase since 1880 (in cm)',
                         'gas oil coal(in TWh)']


def main():
    data_path = config['data_path']
    output_dir_biv = config['output_dirs']['bivariate']
    output_dir_eda = config['output_dirs']['descriptive']
    output_dir_regress = config['output_dirs']['regression']

    dependent_variable = config['dependent_variable']
    weather_events = config['weather_events']

    bivariate_analysis(data_path, output_dir_biv, dependent_variable)
    exploratory_data_analysis(data_path, output_dir_eda, dependent_variable, weather_events,
                              config['independent_variables'])
    regression_summary(data_path, dependent_variable, config['independent_variables'], output_dir_regress)

    for main_variable in weather_events:
        regression_summary(data_path, main_variable, config['independent_variables'], output_dir_regress)

    # todo: future prognosis


if __name__ == "__main__":
    main()

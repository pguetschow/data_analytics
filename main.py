import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from config import config
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression


def prepare_output_directory(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def descriptive_analysis(data, variable, output_dir):
    desc_stats = data[variable].describe()
    print(f"Descriptive Statistics for {variable}:\n{desc_stats}")

    plt.figure(figsize=(10, 6))
    sns.histplot(data[variable], kde=True, color="blue", stat="density")
    plt.axvline(
        desc_stats["25%"], color="g", linestyle="--", label="25th Percentile (Q1)"
    )
    plt.axvline(
        desc_stats["50%"],
        color="b",
        linestyle="-",
        label="50th Percentile (Q2, Median)",
    )
    plt.axvline(
        desc_stats["75%"], color="g", linestyle="--", label="75th Percentile (Q3)"
    )
    plt.legend()
    plt.title(f"Histogram with Bell Curve of {variable}")
    plt.xlabel(variable)
    plt.ylabel("Density")
    plt.savefig(os.path.join(output_dir, f"histogram_bell_curve_{variable}.png"))
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[variable], color="lightblue")
    plt.title(f"Boxplot of {variable}")
    plt.savefig(os.path.join(output_dir, f"boxplot_{variable}.png"))
    plt.show()


def plot_multiple_events_over_time(data, event_columns, output_dir):
    plt.figure(figsize=(10, 6))
    for event in event_columns:
        plt.plot(data["Year"], data[event], label=event)
    plt.title("Trend for Extreme Weather Events over Time")
    plt.xlabel("Year")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "all_events_trend.png"))
    plt.show()


def plot_correlation_matrix(data, selected_columns, output_dir):
    selected_df = data[selected_columns]
    correlation_matrix = selected_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        cbar=True,
        square=True,
        fmt=".2f",
        annot_kws={"size": 10},
    )
    plt.title("Correlation Matrix of Selected Variables")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.show()


def plot_time_series(time, series, title, xlabel, ylabel, window, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(time, series, label="Actual")

    if window:
        rolling_mean = series.rolling(window=window).mean()
        plt.plot(
            time, rolling_mean, color="red", label=f"Rolling Mean (window={window})"
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    plot_filename = f"time_series_{title}.png"
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.show()


def bivariate_analysis(data_path, output_dir, dependent_variable):
    output_file_name = "bivariate_analysis_grand_total.txt"
    df = pd.read_csv(data_path)
    prepare_output_directory(output_dir)
    output_path = os.path.join(output_dir, output_file_name)

    with open(output_path, "w") as file:
        for col in df.columns:
            if col not in IGNORE_COLUMNS:
                file.write(f"Bivariate Analysis for {dependent_variable} and {col}:\n")
                plt.figure(figsize=(8, 5))
                sns.scatterplot(data=df, x=dependent_variable, y=col)
                plt.title(f"Scatter Plot of {dependent_variable} and {col}")
                plot_filename = f"scatterplot_{dependent_variable}_{col}.png"
                plt.savefig(os.path.join(output_dir, plot_filename))
                plt.close()
                correlation = df[dependent_variable].corr(df[col])
                file.write(f"Correlation Coefficient: {correlation:.2f}\n\n")
                covariance = df[dependent_variable].cov(df[col])
                file.write(f"Covariance: {covariance:.2f}\n\n")
                file.write("-" * 50 + "\n\n")

                sns.regplot(x=dependent_variable, y=col, data=df)
                plt.title(f"Regression Plot of {dependent_variable} and {col}")
                plot_filename = f"regplot_{dependent_variable}_{col}.png"
                plt.savefig(os.path.join(output_dir, plot_filename))
                plt.close()


def exploratory_data_analysis(
    data_path, output_dir, main_variable, event_columns, independent_variables
):
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
        if col not in ["Year", independent_variables]:
            plot_time_series(
                df["Year"],
                df[col],
                f"Time-Series of {col}",
                "Year",
                col,
                window=5,
                output_dir=output_dir,
            )


def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variables"] = X.columns
    vif_data["VIF"] = [compute_single_vif(X, variable) for variable in X.columns]
    return vif_data


def compute_single_vif(X, target_variable):
    # Split variables into target and predictors
    x_target = X[target_variable]
    x_predictors = X.drop(columns=[target_variable])

    # Add a constant to the predictors matrix
    x_predictors = sm.add_constant(x_predictors)

    # Fit the linear regression model
    model = sm.OLS(x_target, x_predictors).fit()

    # Extract R-squared value
    r_squared = model.rsquared

    # Calculate VIF
    vif = 1 / (1 - r_squared)
    return vif


def print_multicollinearity(X, threshold=5.0):
    vif_data = calculate_vif(X)
    high_vif = vif_data[vif_data["VIF"] > threshold]
    if not high_vif.empty:
        print("Warning: Potential multicollinearity detected:")
        print(high_vif)


def fit_regression_model(X, y):
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit()


def analyze_regression_model_results(model, file, alpha=0.05):
    results_summary = model.summary2().tables[1]

    file.write("Automated Analysis of the Regression Results\n")
    file.write("-" * 45 + "\n")

    rsquared = model.rsquared
    file.write(f"R-squared: {rsquared:.3f}\n")
    if rsquared >= 0.7:
        file.write(
            "The model explains a significant amount of the response variable's variability.\n"
        )
    elif rsquared >= 0.5:
        file.write(
            "The model explains a moderate amount of the response variable's variability.\n"
        )
    else:
        file.write(
            "The model does not explain much of the response variable's variability.\n"
        )

    significant_vars = results_summary[results_summary["P>|t|"] <= alpha].index.tolist()
    if "const" in significant_vars:
        significant_vars.remove("const")

    if significant_vars:
        file.write(
            f"\nSignificant predictors at alpha={alpha}: {', '.join(significant_vars)}.\n"
        )
    else:
        file.write(f"\nNo significant predictors found at alpha={alpha}.\n")

    file.write("-" * 45 + "\n")


def regression_summary(
    data_path, dependent_variable, independent_variables, output_dir
):
    dependent_variable_clean = dependent_variable.replace(" ", "_")
    output_file_name = f"linear_regression_analysis_{dependent_variable_clean}.txt"

    df = pd.read_csv(data_path)
    prepare_output_directory(output_dir)
    output_path = os.path.join(output_dir, output_file_name)

    X = df[independent_variables]
    print_multicollinearity(X)

    y = df[dependent_variable]
    model = fit_regression_model(X, y)

    with open(output_path, "w") as file:
        file.write(model.summary().as_text() + "\n")
        analyze_regression_model_results(model, file)


def analyze_ridge_model_results(model, scaler, independent_variables, file, alpha=1.0):
    file.write("Automated Analysis of the Ridge Regression Results\n")
    file.write("-" * 50 + "\n")

    # Explanation of Ridge Regression and the role of alpha
    file.write(
        "Ridge Regression is a type of linear regression that includes a regularization term. The regularization term discourages overly complex models which can lead to overfitting. The strength of the regularization is controlled by the parameter alpha.\n"
    )

    # Alpha interpretation
    file.write(f"Value of Alpha (Regularization Strength): {alpha}\n")
    if alpha == 0:
        file.write(
            "This is equivalent to ordinary least squares regression without regularization.\n"
        )
    elif 0 < alpha < 1:
        file.write(
            "A small value of alpha implies minimal regularization. The model relies mostly on the data and less on the regularization term.\n"
        )
    elif alpha == 1:
        file.write(
            "Balanced regularization applied. The model equally relies on the data and the regularization term.\n"
        )
    else:
        file.write(
            "High value of alpha. Strong regularization applied which may dominate the objective function, leading to coefficient shrinkage and a simpler model.\n"
        )

    # Bias/Intercept Explanation
    file.write("\nIntercept (Bias): {:.4f}\n".format(model.intercept_))
    file.write(
        "The intercept represents the predicted value of the dependent variable when all independent variables are at their mean value (because of standardization). In non-standardized scenarios, it would be the predicted value when all predictors have a value of zero, though this might not be meaningful for all variables.\n"
    )

    standardized_coeffs = model.coef_
    original_scale_coeffs = standardized_coeffs / scaler.scale_

    for var, coef, original_coef in zip(
        independent_variables, standardized_coeffs, original_scale_coeffs
    ):
        file.write(f"\nVariable: {var}\n")
        file.write(f"Coefficient (Standardized Scale): {coef:.4f}\n")
        file.write(f"Coefficient (Original Scale): {original_coef:.4f}\n")

        # Enhanced Interpretation of coefficients
        if abs(coef) > 1.0:
            file.write(
                f"The variable '{var}' has a strong impact on the dependent variable in its standardized form.\n"
            )
            if original_coef > 0:
                file.write(
                    f"For every one unit increase in '{var}' (in its original scale), the dependent variable is expected to increase by approximately {original_coef:.4f}, keeping other factors constant.\n"
                )
            else:
                file.write(
                    f"For every one unit increase in '{var}' (in its original scale), the dependent variable is expected to decrease by approximately {original_coef:.4f}, keeping other factors constant.\n"
                )
        else:
            file.write(
                f"The variable '{var}' has a moderate to low impact on the dependent variable in its standardized form.\n"
            )

    # Insights into relative importance
    absolute_coeffs = [abs(coeff) for coeff in standardized_coeffs]
    total_importance = sum(absolute_coeffs)
    relative_importances = [
        (var, abs(coeff) / total_importance)
        for var, coeff in zip(independent_variables, standardized_coeffs)
    ]
    sorted_importances = sorted(relative_importances, key=lambda x: x[1], reverse=True)

    file.write("\nRelative Importance of Predictors (Standardized):\n")
    for var, imp in sorted_importances:
        file.write(f"{var}: {imp * 100:.2f}% of total standardized importance.\n")

    file.write("-" * 50 + "\n")


def fit_ridge_regression_model(X, y, alphas):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
    ridge_cv.fit(X_scaled, y)

    best_alpha = ridge_cv.alpha_  # Get the best alpha value
    return ridge_cv, scaler, best_alpha


def ridge_summary(
    data_path,
    dependent_variable,
    independent_variables,
    output_dir,
    alphas=[0.25, 0.5, 0.75, 1],
):
    dependent_variable_clean = dependent_variable.replace(" ", "_")
    output_file_name = f"ridge_regression_analysis_{dependent_variable_clean}.txt"

    df = pd.read_csv(data_path)
    prepare_output_directory(output_dir)
    output_path = os.path.join(output_dir, output_file_name)

    X = df[independent_variables]
    y = df[dependent_variable]

    model, scaler, best_alpha = fit_ridge_regression_model(X, y, alphas=alphas)

    with open(output_path, "w") as file:
        file.write(
            f"Ridge Regression Analysis (best alpha={best_alpha}) for {dependent_variable}\n"
        )
        file.write("-" * 45 + "\n")
        file.write(f"Intercept: {model.intercept_:.4f}\n\n")
        analyze_ridge_model_results(
            model, scaler, independent_variables, file, best_alpha
        )


def fit_pls_regression_model(X, y, num_components):
    pls = PLSRegression(n_components=num_components)
    pls.fit(X, y)
    return pls


def analyze_pls_model_results(model, independent_variables, file):
    file.write("Automated Analysis of the Partial Least Squares Regression Results\n")
    file.write("-" * 50 + "\n")

    # Intercept interpretation
    file.write(f"Intercept (Bias): {model._y_mean[0]:.4f}\n")
    file.write(
        "The intercept represents the predicted value of the dependent variable when all independent variables are at their mean values.\n"
    )

    for var, coef in zip(independent_variables, model.coef_):
        file.write(f"\nVariable: {var}\n")
        file.write(f"Coefficient: {coef[0]:.4f}\n")

        # Enhanced Interpretation of coefficients
        if abs(coef[0]) > 1.0:
            if coef[0] > 0:
                file.write(
                    f"For every one unit increase in '{var}', the dependent variable is expected to increase by approximately {coef[0]:.4f}, keeping other factors constant.\n"
                )
            else:
                file.write(
                    f"For every one unit increase in '{var}', the dependent variable is expected to decrease by approximately {coef[0]:.4f}, keeping other factors constant.\n"
                )
        else:
            file.write(
                f"The variable '{var}' has a moderate to low impact on the dependent variable.\n"
            )

    # Explaining the variance captured by the PLS model
    explained_variance = sum(model.y_loadings_**2)
    file.write(f"\nExplained Variance by PLS components: {explained_variance:.2f}\n")
    file.write(
        "The explained variance represents the proportion of the total variance in the dependent variable that is captured by the PLS model.\n"
    )

    file.write("-" * 50 + "\n")


def pls_summary(
    data_path, dependent_variable, independent_variables, output_dir, num_components=2
):
    dependent_variable_clean = dependent_variable.replace(" ", "_")
    output_file_name = f"pls_regression_analysis_{dependent_variable_clean}.txt"

    df = pd.read_csv(data_path)
    prepare_output_directory(output_dir)
    output_path = os.path.join(output_dir, output_file_name)

    X = df[independent_variables]
    y = df[dependent_variable]

    model = fit_pls_regression_model(X, y, num_components)

    with open(output_path, "w") as file:
        file.write(
            f"Partial Least Squares Regression Analysis for {dependent_variable}\n"
        )
        file.write("-" * 45 + "\n")
        file.write(f"Number of Components: {num_components}\n")

        # Intercept
        # file.write(f"\nIntercept: {model.getm:.4f}\n")
        print(model)
        # Coefficients
        for var, coef in zip(independent_variables, model.coef_):
            file.write(f"\nVariable: {var}\n")
            file.write(f"Coefficient: {coef[0]:.4f}\n")
            if coef[0] > 0:
                file.write(
                    f"For every one unit increase in '{var}', the dependent variable is expected to increase by approximately {coef[0]:.4f}, keeping other factors constant.\n"
                )
            else:
                file.write(
                    f"For every one unit increase in '{var}', the dependent variable is expected to decrease by approximately {coef[0]:.4f}, keeping other factors constant.\n"
                )

        # Variance Explained
        y_variance_explained = 100 * model.y_scores_.var(axis=0).sum() / y.var()
        file.write(
            f"\nVariance Explained in {dependent_variable}: {y_variance_explained:.2f}%\n"
        )

        # Loadings
        file.write("\nLoadings for each component:\n")
        for i, component in enumerate(model.x_loadings_.T):
            file.write(f"\nComponent {i + 1}:\n")
            for var, loading in zip(independent_variables, component):
                file.write(f"{var}: {loading:.4f}\n")

        # Importance of Predictors
        importance = (model.x_weights_**2).sum(axis=1)
        file.write("\nImportance of Predictors:\n")
        for var, imp in zip(independent_variables, importance):
            file.write(f"{var}: {imp:.4f}\n")

        file.write("-" * 45 + "\n")


# Constants
IGNORE_COLUMNS = [
    "Year",
    "Grand Total of Extreme Weather Events",
    "Flood",
    "Extreme weather",
    "Drought",
    "Extreme temperature",
    "Wildfire",
]
INDEPENDENT_VARIABLES = config["independent_variables"]


def main():
    data_path = config["data_path"]
    output_dir_biv = config["output_dirs"]["bivariate"]
    output_dir_eda = config["output_dirs"]["descriptive"]
    output_dir_regress = config["output_dirs"]["regression"]
    output_dir_prognosis = config["output_dirs"]["prognosis"]
    dependent_variable = config["dependent_variable"]
    weather_events = config["weather_events"]

    bivariate_analysis(data_path, output_dir_biv, dependent_variable)
    exploratory_data_analysis(
        data_path,
        output_dir_eda,
        dependent_variable,
        weather_events,
        config["independent_variables"],
    )

    regression_summary(
        data_path,
        dependent_variable,
        config["independent_variables"],
        output_dir_regress,
    )
    ridge_summary(
        data_path,
        dependent_variable,
        config["independent_variables"],
        output_dir_regress,
    )
    pls_summary(
        data_path,
        dependent_variable,
        config["independent_variables"],
        output_dir_regress,
    )

    # todo: future prognosis


if __name__ == "__main__":
    main()

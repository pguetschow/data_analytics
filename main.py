import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import config
from scipy.stats import chi2_contingency, pearsonr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
import statsmodels.api as sm
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.formula.api import ols
from scipy import stats
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

CATEGORICAL_VARIABLES = ['AgeCategory', 'HighBP', 'HighChol', 'Smoker', 'Stroke', 'Diabetes',
                         'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
                         'NoDocbcCost', 'DiffWalk', 'Sex', 'Education', 'Income']

BINARY_VARIABLES = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'Diabetes', 'PhysActivity',
                    'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
                    'DiffWalk', 'Sex']

NUMERIC_VARIABLES = ['BMI']


def prepare_output_directory(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def exploratory_data_analysis(df, output_dir, dependent_variable, independent_variables):
    desc = df.describe()
    desc.to_csv(os.path.join(output_dir, "summary_statistics.csv"))

    correlation = df[independent_variables].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, cmap="coolwarm")
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()


def bivariate_analysis(df, output_dir, dependent_variable):
    for column in df.columns:
        if column != dependent_variable:
            plt.figure(figsize=(10, 6))
            if column in CATEGORICAL_VARIABLES:
                sns.countplot(data=df, hue=dependent_variable, x=column)
                plt.title(f"{column} vs {dependent_variable}")
                plt.tight_layout()
            else:
                sns.histplot(data=df, x=column, hue=dependent_variable, kde=True)
                plt.xlabel(column)
                plt.ylabel('Frequency')

            plt.savefig(os.path.join(output_dir, f'{column}.png'))
            plt.close()


def correlation_test(df, continuous_variables, dependent_variable):
    correlations = {col: pearsonr(df[col], df[dependent_variable]) for col in continuous_variables}
    return pd.DataFrame(correlations, index=['correlation_coeff', 'p_value'])


def chi_square_test(df, categorical_variables, dependent_variable):
    chi2_values = {
        col: chi2_contingency(pd.crosstab(df[col], df[dependent_variable]))[:2]
        for col in categorical_variables
    }
    return pd.DataFrame(chi2_values, index=['chi2_stat', 'p_value'])


def interpret_regression_results(regression_result, independent_variables):
    interpretation = ""

    # Intercept interpretation
    interpretation += f"The intercept of the model is {regression_result['Intercept']}. This represents the predicted value of the dependent variable when all independent variables are set to 0.\n\n"

    # Coefficients interpretation
    interpretation += "Coefficients Interpretation:\n"
    for var, coef in zip(independent_variables, regression_result['Coefficients']):
        interpretation += f"For every one-unit increase in {var}, the dependent variable is expected to change by {coef} units, holding all other variables constant.\n"

    # MSE interpretation
    interpretation += f"\nMean Squared Error of the model is {regression_result['Mean Squared Error']}. A lower MSE indicates a model that fits the data more closely."

    return interpretation

# Call the interpret_regression_results function after saving the results to CSV
def regression_analysis(df, dependent_variable, independent_variables, output_dir):
    X, y = df[independent_variables], df[dependent_variable]
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    regression_result = {
        "Coefficients": model.coef_,
        "Intercept": model.intercept_,
        "Mean Squared Error": mean_squared_error(y, y_pred)
    }

    # Save to CSV
    pd.DataFrame(regression_result).to_csv(os.path.join(output_dir, "regression_results.csv"))

    # Interpret results
    interpretation = interpret_regression_results(regression_result, independent_variables)
    with open(os.path.join(output_dir, "regression_interpretation.txt"), "w") as file:
        file.write(interpretation)


def check_multicollinearity(df, independent_variables):
    X = sm.add_constant(df[independent_variables])
    vif_data = {
        "Variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }

    return pd.DataFrame(vif_data)



def check_anova_assumptions(residuals):
    """Checks basic assumptions for ANOVA."""
    # Check normality of residuals
    w, p_value_normality = stats.shapiro(residuals)
    normality_str = f"Residuals appear to be normally distributed (Shapiro-Wilk p={p_value_normality:.5f})." \
        if p_value_normality > 0.05 else \
        f"Residuals might not be normally distributed (Shapiro-Wilk p={p_value_normality:.5f})."

    # Other assumptions such as homoscedasticity can be added here.

    return normality_str


def interpret_anova_results(anova_table, alpha=0.05):
    df_between = anova_table["df"].iloc[0]
    df_within = anova_table["df"].iloc[1]
    F_value = anova_table["F"].iloc[0]
    p_val = anova_table["PR(>F)"].iloc[0]

    interpretation = f"Between Groups Degrees of Freedom (df): {df_between}\n" \
                     f"Within Groups Degrees of Freedom (df): {df_within}\n" \
                     f"F-value: {F_value:.4f}\n" \
                     f"p-value: {p_val:.5f}\n"
    significance_str = f"The results of the ANOVA test are statistically significant (p={p_val:.5f}). There are differences in the group means." \
        if p_val < alpha else \
        f"The results of the ANOVA test are not statistically significant (p={p_val:.5f}). There might not be significant differences in the group means."
    return interpretation + significance_str


def conduct_anova(df, independent_var, dependent_var):
    formula = f"{dependent_var} ~ C({independent_var})"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table, model.resid

def multivariate_categorical_analysis(df, independent_variables, dependent_variable, output_dir):
    output_path = os.path.join(output_dir, "multivariate_categorical_analysis.txt")

    with open(output_path, "w") as file:
        file.write("\nANOVA Results\n" + "-" * 45 + "\n")
        for var in independent_variables:
            if df[var].nunique() > 2:
                anova_results, residuals = conduct_anova(df, var, dependent_variable)
                assumptions_check = check_anova_assumptions(residuals)
                interpretation = interpret_anova_results(anova_results)
                file.write(f"ANOVA for {var}:\n{anova_results.to_string()}\n{interpretation}\n{assumptions_check}\n\n")


def logistic_regression_analysis(df, dependent_variable, independent_variables, output_dir):
    X, y = df[independent_variables], df[dependent_variable]
    model = LogisticRegression(max_iter=1000).fit(X, y)
    y_pred = model.predict(X)

    regression_df = pd.DataFrame({
        "Coefficients": model.coef_[0],
        "Intercept": model.intercept_[0],
    })
    regression_df.to_csv(os.path.join(output_dir, "logistic_regression_results.csv"))

    report_df = pd.DataFrame(classification_report(y, y_pred, output_dict=True)).transpose()
    report_df.to_csv(os.path.join(output_dir, "logistic_regression_classification_report.csv"))

    cm_df = pd.DataFrame(confusion_matrix(y, y_pred), columns=[f"Predicted_{c}" for c in y.unique()], index=[f"Actual_{c}" for c in y.unique()])
    cm_df.to_csv(os.path.join(output_dir, "logistic_regression_confusion_matrix.csv"))

    return model


def discriminant_analysis(df, dependent_variable, independent_variables, output_dir):
    X, y = df[independent_variables], df[dependent_variable]
    lda = LinearDiscriminantAnalysis().fit(X, y)
    y_pred = lda.predict(X)

    coefficients_df = pd.DataFrame({
        "Coefficients": lda.coef_[0],
        "Intercept": np.repeat(lda.intercept_, len(lda.coef_[0])),
    })
    coefficients_df.to_csv(os.path.join(output_dir, "discriminant_analysis_results.csv"))

    report_df = pd.DataFrame(classification_report(y, y_pred, output_dict=True)).transpose()
    report_df.to_csv(os.path.join(output_dir, "discriminant_analysis_classification_report.csv"))

    cm_df = pd.DataFrame(confusion_matrix(y, y_pred), columns=[f"Predicted_{c}" for c in y.unique()], index=[f"Actual_{c}" for c in y.unique()])
    cm_df.to_csv(os.path.join(output_dir, "discriminant_analysis_confusion_matrix.csv"))

    return lda


def main():
    data_path = config["data_path"]
    output_dir_eda = config["output_dirs"]["descriptive"]
    output_dir_biv = config["output_dirs"]["bivariate"]
    output_dir_regress = config["output_dirs"]["regression"]
    output_dir_anova = config["output_dirs"]["anova"]

    prepare_output_directory(output_dir_eda)
    prepare_output_directory(output_dir_biv)
    prepare_output_directory(output_dir_regress)
    prepare_output_directory(output_dir_anova)

    df = pd.read_csv(data_path)
    dependent_variable = config["dependent_variable"]

    exploratory_data_analysis(df, output_dir_eda, dependent_variable, config["independent_variables"])
    bivariate_analysis(df, output_dir_biv, dependent_variable)
    correlation_results = correlation_test(df, NUMERIC_VARIABLES, dependent_variable)
    correlation_results.to_csv(os.path.join(output_dir_eda, "correlation_test_results.csv"))
    chi2_results = chi_square_test(df, CATEGORICAL_VARIABLES, dependent_variable)
    chi2_results.to_csv(os.path.join(output_dir_eda, "chi_square_test_results.csv"))
    regression_analysis(df, dependent_variable, NUMERIC_VARIABLES, output_dir_regress)
    multivariate_categorical_analysis(df, CATEGORICAL_VARIABLES, dependent_variable, output_dir_anova)

    logistic_regression_analysis(df, dependent_variable, BINARY_VARIABLES, output_dir_regress)
    discriminant_analysis(df, dependent_variable, BINARY_VARIABLES, output_dir_regress)


if __name__ == "__main__":
    main()

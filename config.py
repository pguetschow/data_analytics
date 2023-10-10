config = {
    'data_path': 'data/data_full.csv',
    # 'data_path': 'data/sampled_data.csv',
    'output_dirs': {
        'bivariate': 'export/bivariate',
        'descriptive': 'export/descriptive',
        'regression': 'export/regression',
        'prognosis': 'export/prognosis',
        'anova': 'export/anova',
    },
    'independent_variables': ['HighBP', 'HighChol', 'Smoker', 'Diabetes', 'PhysActivity', 'HvyAlcoholConsump',
                              'AnyHealthcare', 'NoDocbcCost', 'Sex', 'AgeCategory', 'Education', 'Income']
    ,
    'dependent_variable': 'HeartDiseaseorAttack'
}

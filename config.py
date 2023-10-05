config = {
    'data_path': 'data/data.csv',
    'output_dirs': {
        'bivariate': 'export/bivariate',
        'descriptive': 'export/descriptive',
        'regression': 'export/regression',
        'prognosis': 'export/prognosis',
    },
    'ignore_columns': ['Year', 'Grand Total', 'Flood', 'Extreme weather', 'Drought', 'Extreme temperature', 'Wildfire'],
    'independent_variables': [
        'Temperature Anomalies in Celsius',
        'Average CO2 levels in the atmosphere worldwide(in ppm)',
        'Annual CO2 emissions worldwide (in billion metric tons)',
        # 'Adjusted sea level increase since 1880 (in cm)',
        'gas oil coal(in TWh)'
    ],
    'weather_events': ['Flood', 'Extreme weather', 'Drought', 'Extreme temperature', 'Wildfire'],
    'dependent_variable': 'Grand Total'
}

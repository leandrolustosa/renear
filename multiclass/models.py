models = [

    # { 'case': 'fish', 'path': '../resources/fish/dataset.csv', 
    #   'X_cols': ['Weight', 'Width'], 'y_col': 'Species', 'cols_remove': [], 
    #   'X_train': None, 'y_train': None, 'X_test': None, 'y_test': None,
    #   'epochs': 10000, 'alpha': 1e-2, 'threshold': 0.9, 'is_binary': False },

    { 'case': 'diabetes', 'path': '../resources/diabetes/dataset.csv', 
      'X_cols': ['Glucose', 'BloodPressure'], 'y_col': 'Outcome', 'cols_remove': [],
      'X_train': None, 'y_train': None, 'X_test': None, 'y_test': None,
      'epochs': 10000, 'alpha': 1e-2, 'threshold': 0.9, 'is_binary': True }

    # { 'case': 'mushrooms', 'path': '../resources/mushrooms/dataset.csv', 
    #   'X_cols': ['gill-color', 'spore-print-color'], 'y_col': 'class', 'cols_remove': ['veil-type'],
    #   'X_train': None, 'y_train': None, 'X_test': None, 'y_test': None,
    #   'epochs': 10000, 'alpha': 1e-2, 'threshold': 0.9, 'is_binary': True },

    # { 'case': 'heart_disease', 'path': '../resources/heart_disease/dataset.csv', 
    #   'X_cols': ['calories_daily', 'exercise_daily'], 'y_col': 'target', 'cols_remove': ['id', 'thal', 'ca', 'slope', 'exang', 'restecg', 'fbs', 'cp', 'trestbps', 'age', 'chol', 'oldpeak', 'health', 'fat', 'env', 'cra'],
    #   'X_train': None, 'y_train': None, 'X_test': None, 'y_test': None,
    #   'epochs': 10000, 'alpha': 1e-2, 'threshold': 0.9, 'is_binary': True }

]
import pandas as pd
from application_logging import logger
from data_preprocessing import preprocessing
from Pickle import pickle


class Train_Model:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        filename = 'Training_Logs/ModelTrainingLogs.txt'
        with open(filename, 'w') as file:
            pass
        self.file_object = open('Training_Logs/ModelTrainingLogs.txt', "a+")

    def train_model(self):
        self.log_writer.log(self.file_object, "Start of Training")

        try:
            preprocessor = preprocessing.Preprocessor(
                self.file_object, self.log_writer)

            train_df = preprocessor.load_data("train.csv")
            print("Sample Dataset: ")
            print(train_df.head(10))

            # Check for categorical variables
            categorical_variables = preprocessor.check_categorical_variables(
                train_df)
            print("Categorical Features: ")
            print(categorical_variables)

            # Convert selected columns to float64
            train_df = preprocessor.convert_columns_to_float(train_df)

            # Rename selected columns
            train_df = preprocessor.rename_columns(train_df)

            # Remove specified columns
            train_df = preprocessor.remove_columns(train_df)
            print("Dataset after converting 'dependency', 'edjefe', 'edjefa' feature to float and removing Squre feature: ")
            print(train_df)

            # Plot column distributions
            print("feature Distribution: ")
            # preprocessor.plot_column_distribution(train_df)

            # combine float and int
            df_combined = preprocessor.combine_int_float(train_df)
            print("Float and int feature: ")
            print(df_combined)

            result_df = preprocessor.feature_imputation(train_df, df_combined)
            print("Minimum mean squre error for each feature: ")
            print(result_df)

            result_df = preprocessor.feature_imputation_verification(
                df_combined)
            print("Minimum mean squre error for selected features for verification: ")
            print(result_df)

            train_df = preprocessor.replace_null_with_mean(train_df)
            print("After replacing Null values with mean: ")
            preprocessor.check_null_values(train_df)

            diff_target_value, same_target_value = preprocessor.check_same_poverty_level(
                train_df)
            family_head = preprocessor.find_family_head(train_df)

            families_without_head = preprocessor.check_families_without_head(
                train_df, family_head)
            families_without_head_same = preprocessor.check_families_without_head_same(
                train_df, families_without_head)

            train_df = preprocessor.set_same_poverty_level(
                train_df, diff_target_value)
            preprocessor.check_same_poverty_level(train_df)
            print("Checking Null values in target: ")
            preprocessor.check_null_values_in_target(train_df)

            train_df = train_df.drop('idhogar', axis=1)

            preprocessor.select_top_30_features_with_highest_variance(train_df)
            preprocessor.handle_outliers_using_winsorization(train_df)

            preprocessor.check_feature_imbalance(train_df)
            preprocessor.handle_feature_imbalance(train_df)
            print("Distribution of Target Feature: ")
            preprocessor.plot_class_distribution(train_df)
            top_features = preprocessor.get_top_correlated_features(train_df)
            print("Top 10 Features with highest correlation")
            print(top_features)

            relevant_features = preprocessor.get_relevant_features(train_df)
            print("Features having correlation of threshold 0.18")
            print(relevant_features)

            print("Selecting Model: ")
            preprocessor.fit_and_evaluate_models(train_df)

            accuracy = preprocessor.train_random_forest_classifier(train_df)
            print("Accuracy for Random Forest Classifier: ")
            print('Accuracy:', "{:.2f}".format(accuracy))

            # Get the top features with highest variance
            top_30_feature_names = preprocessor.get_top_features_with_highest_variance(
                train_df)
            print("Top 30 features with highest variance: ")
            print(top_30_feature_names)

            print("Checking Which features are missing for different threshold values: ")
            preprocessor.check_missing_features(train_df, top_30_feature_names)

            print("Top 6 thresholds with highest Accuracy: ")
            preprocessor.find_top_thresholds(train_df, top_30_feature_names)

            print("Checking Accuracy for each threshold level with adding missing features with highest variance: ")
            preprocessor.find_thresholds_accuracy(
                train_df, top_30_feature_names)

            print(
                "Predicting the accuracy for Random Forest Classifier using Hyperparameter tuning")

            # preprocessor.hyperparameter_tuning(train_df, top_30_feature_names)

            print(
                "Predicting the accuracy for Random Forest Classifier usin Cross validation using Kfold")
            preprocessor.cross_validation(train_df)

            print("Confusion matrix and Classification report")
            best_model_name, best_model = preprocessor.generate_report(
                train_df, top_30_feature_names)

            # saving the best model to the directory.
            file_op = pickle.File_Operation(
                self.file_object, self.log_writer)
            save_model = file_op.save_model(best_model, best_model_name)

            self.log_writer.log(self.file_object, "Successful end of training")

        except Exception as e:
            self.log_writer.log(
                self.file_object, "Unsuccessful end of training")
            raise Exception()

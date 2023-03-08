import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import mean_squared_error


class Preprocessor:

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def load_data(self, file_path):
        """
        Loads data from a CSV file and returns a pandas DataFrame.
        """
        self.logger_object.log(
            self.file_object, "Entered the load_data method in Preprocessor class.")
        return pd.read_csv(file_path)

    def check_categorical_variables(self, data):
        """
        Checks for categorical features in a pandas DataFrame and returns a list of column names.
        """
        self.logger_object.log(
            self.file_object, "Entered the check_categorical_variables method in Preprocessor class.")
        try:
            categorical_variables = []
            for col in data.columns:
                if data[col].dtype == 'object':
                    categorical_variables.append(col)
            return categorical_variables
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in check_categorical_variables method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def convert_columns_to_float(self, data):
        """
        Converts the data type of specified columns in a pandas DataFrame to float64.
        If a mapping is provided, replaces values in the specified columns with values from the mapping before conversion.
        """
        self.logger_object.log(
            self.file_object, "Entered the convert_columns_to_float method in Preprocessor class.")
        try:
            mapping = {'yes': 1, 'no': 0}
            columns_to_convert = ['dependency', 'edjefe', 'edjefa']
            if mapping is not None:
                for col in columns_to_convert:
                    data[col] = data[col].replace(mapping)
            data[columns_to_convert] = data[columns_to_convert].astype(
                np.float64)
            return data
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in convert_columns_to_float method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def rename_columns(self, data):
        """
        Renames specified columns in a pandas DataFrame using a dictionary of column name mappings.
        """
        self.logger_object.log(
            self.file_object, "Entered the rename_columns method in Preprocessor class.")
        try:
            column_mapping = {'dependency': '_dependency',
                              'edjefe': '_edjefe', 'edjefa': '_edjefa'}
            return data.rename(columns=column_mapping)
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in rename_columns method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def remove_columns(self, data):
        """
        Removes specified columns from a pandas DataFrame.
        """
        self.logger_object.log(
            self.file_object, "Entered the remove_columns method in Preprocessor class.")
        try:
            return data.drop(['Id', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin',
                              'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq'], axis=1)
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in remove_columns method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def plot_column_distribution(self, data):
        """
        Plots a histogram for each column in a pandas DataFrame.
        """
        self.logger_object.log(
            self.file_object, "Entered the plot_column_distribution method in Preprocessor class.")
        try:
            for column in data.columns:
                plt.hist(data[column])
                plt.xlabel(column)
                plt.ylabel("Frequency")
                plt.title("Distribution of " + column)
                plt.show()
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in plot_column_distribution method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def combine_int_float(self, data):
        """
        Separates a pandas DataFrame into two new DataFrames: one for columns of data type float64,
        and one for columns of data type int64. Returns a new DataFrame with missing values imputed.
        """
        self.logger_object.log(
            self.file_object, "Entered the combine_int_float method in Preprocessor class.")
        try:
            # Get columns with float64 and int64 data types
            float_cols = data.select_dtypes('float64').columns
            int_cols = data.select_dtypes('int64').columns

            # Combine the two column types
            combined_cols = list(float_cols) + list(int_cols)

            # Create new dataframe with combined columns
            df_combined = data[combined_cols]

            return df_combined
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in combine_int_float method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def feature_imputation(self, train_df, df_combined):
        self.logger_object.log(
            self.file_object, "Entered the feature_imputation method in Preprocessor class.")
        try:
            result = []

            # Loop through each feature in the list
            for feature in df_combined:
                # Calculate the mean, mode, and median for the feature
                col_mean = np.mean(train_df[feature])
                col_mode = train_df[feature].mode()[0]
                col_median = np.median(train_df[feature])

                # Calculate the mean square error for the mean, mode, and median
                col_mse_mean = np.mean((train_df[feature] - col_mean) ** 2)
                col_mse_mode = np.mean((train_df[feature] - col_mode) ** 2)
                col_mse_median = np.mean((train_df[feature] - col_median) ** 2)

                # Fill in missing values with ffill and calculate the mean square error
                ffill_values = train_df[feature].fillna(method='ffill')
                col_mse_ffill = np.mean(
                    (train_df[feature] - ffill_values) ** 2)

                # Fill in missing values with bfill and calculate the mean square error
                bfill_values = train_df[feature].fillna(method='bfill')
                col_mse_bfill = np.mean(
                    (train_df[feature] - bfill_values) ** 2)

                # Store the results in a dictionary and append it to the result list
                feature_result = {'Feature': feature, 'Mean': col_mean, 'Mode': col_mode, 'Median': col_median,
                                  'MSE Mean': col_mse_mean, 'MSE Mode': col_mse_mode, 'MSE Median': col_mse_median,
                                  'MSE ffill': col_mse_ffill, 'MSE bfill': col_mse_bfill}
                result.append(feature_result)

            # Convert the result list to a dataframe and display the lowest MSE for each feature
            result_df = pd.DataFrame(result)

            # Sort the dataframe by the MSE for each feature
            result_df_sorted = result_df.sort_values(
                by=['MSE Mean', 'MSE Mode', 'MSE Median'])
            result_df['Min MSE column'] = result_df[[
                'MSE Mean', 'MSE Mode', 'MSE Median']].idxmin(axis=1)
            result_df['Min MSE'] = result_df[[
                'MSE Mean', 'MSE Mode', 'MSE Median']].min(axis=1)

            return result_df
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in feature_imputation method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def feature_imputation_verification(self, df_combined):
        self.logger_object.log(
            self.file_object, "Entered the feature_imputation_verification method in Preprocessor class.")
        try:
            features = ['hacdor', 'rooms', 'v18q', 'r4h1', 'r4m1', 'r4m3', 'r4t1', 'escolari', 'paredblolad', 'pisomoscer', 'pisocemento', 'cielorazo', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho3',
                        'eviv1', 'eviv2', 'eviv3', 'hogar_nin', 'meaneduc', 'instlevel8', 'overcrowding', 'computer', 'qmobilephone', 'lugar1', 'v2a1', 'age', 'tamviv', 'tamhog', 'hhsize', 'hogar_total', 'r4t3', 'r4t2']

            result = []

            # Loop through each feature in the list
            for feature in features:
                # Calculate the mean, mode, and median for the feature
                col_mean = np.mean(df_combined[feature])
                col_mode = df_combined[feature].mode()[0]
                col_median = np.median(df_combined[feature])

                # Calculate the mean square error for the mean, mode, and median
                col_mse_mean = np.mean((df_combined[feature] - col_mean) ** 2)
                col_mse_mode = np.mean((df_combined[feature] - col_mode) ** 2)
                col_mse_median = np.mean(
                    (df_combined[feature] - col_median) ** 2)

                # Fill in missing values with ffill and calculate the mean square error
                ffill_values = df_combined[feature].fillna(method='ffill')
                col_mse_ffill = np.mean(
                    (df_combined[feature] - ffill_values) ** 2)

                # Fill in missing values with bfill and calculate the mean square error
                bfill_values = df_combined[feature].fillna(method='bfill')
                col_mse_bfill = np.mean(
                    (df_combined[feature] - bfill_values) ** 2)

                # Store the results in a dictionary and append it to the result list
                feature_result = {'Feature': feature, 'Mean': col_mean, 'Mode': col_mode, 'Median': col_median,
                                  'MSE Mean': col_mse_mean, 'MSE Mode': col_mse_mode, 'MSE Median': col_mse_median,
                                  'MSE ffill': col_mse_ffill, 'MSE bfill': col_mse_bfill}
                result.append(feature_result)

            # Convert the result list to a dataframe and display the lowest MSE for each feature
            result_df = pd.DataFrame(result)

            # Sort the dataframe by the MSE for each feature
            result_df_sorted = result_df.sort_values(
                by=['MSE Mean', 'MSE Mode', 'MSE Median'])
            result_df['Min MSE column'] = result_df[[
                'MSE Mean', 'MSE Mode', 'MSE Median']].idxmin(axis=1)
            result_df['Min MSE'] = result_df[[
                'MSE Mean', 'MSE Mode', 'MSE Median']].min(axis=1)

            return result_df
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in feature_imputation_verification method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def replace_null_with_mean(self, df):
        self.logger_object.log(
            self.file_object, "Entered the replace_null_with_mean method in Preprocessor class.")
        try:
            cols_to_fill = [col for col in df.columns if col != 'idhogar']
            df[cols_to_fill] = df[cols_to_fill].fillna(df[cols_to_fill].mean())
            return df
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in replace_null_with_mean method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def check_same_poverty_level(self, df):
        self.logger_object.log(
            self.file_object, "Entered the check_same_poverty_level method in Preprocessor class.")
        try:
            same_target_value = df.groupby(
                'idhogar')['Target'].apply(lambda x: x.nunique() == 1)
            diff_target_value = same_target_value[same_target_value != True]
            print('{} members of the house have the different poverty level.'.format(
                len(diff_target_value)))
            return diff_target_value, same_target_value
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in check_same_poverty_level method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def find_family_head(self, df):
        self.logger_object.log(
            self.file_object, "Entered the find_family_head method in Preprocessor class.")
        try:
            family_head = df.groupby('idhogar')['parentesco1'].sum()
            return family_head
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in find_family_head method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def check_families_without_head(self, df, family_head):
        self.logger_object.log(
            self.file_object, "Entered the check_families_without_head method in Preprocessor class.")
        try:
            families_without_head = df.loc[df['idhogar'].isin(
                family_head[family_head == 0].index), :]
            print("{} families don't have any family head.".format(
                families_without_head['idhogar'].nunique()))
            return families_without_head
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in check_families_without_head method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def check_families_without_head_same(self, df, families_without_head):
        self.logger_object.log(
            self.file_object, "Entered the check_families_without_head_same method in Preprocessor class.")
        try:
            families_without_head_same = families_without_head.groupby(
                'idhogar')['Target'].apply(lambda x: x.nunique() == 1)
            print('{} families without head have different poverty level.'.format(
                sum(families_without_head_same == False)))
            return families_without_head_same
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in check_families_without_head_same method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def set_same_poverty_level(self, df, diff_target_value):
        self.logger_object.log(
            self.file_object, "Entered the set_same_poverty_level method in Preprocessor class.")
        try:
            for families in diff_target_value.index:
                head_target = int(df[(df['idhogar'] == families) &
                                     (df['parentesco1'] == 1.0)]['Target'])
                df.loc[df['idhogar'] == families, 'Target'] = head_target
            same_target_value = df.groupby(
                'idhogar')['Target'].apply(lambda x: x.nunique() == 1)
            diff_target_value = same_target_value[same_target_value != True]
            print('{} families with different poverty level.'.format(
                len(diff_target_value)))
            return df
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in set_same_poverty_level method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def check_null_values(self, df):
        self.logger_object.log(
            self.file_object, "Entered the check_null_values method in Preprocessor class.")
        try:
            print('Null values in dataset:', df.isnull().sum().sum())
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in check_null_values method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def check_null_values_in_target(self, df):
        self.logger_object.log(
            self.file_object, "Entered the check_null_values_in_target method in Preprocessor class.")
        try:
            print('Null values in Target column:', df['Target'].isnull().sum())
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in check_null_values_in_target method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def select_top_30_features_with_highest_variance(self, df):
        self.logger_object.log(
            self.file_object, "Entered the select_top_30_features_with_highest_variance method in Preprocessor class.")
        try:
            """
            Selects the top 30 features with highest variance.
            """
            variances = np.var(df, axis=0, ddof=1)
            top_30_indices = np.argsort(variances)[::-1][:30]
            top_30_feature_names = np.array(df.columns)[top_30_indices]
            top_30_variances = variances[top_30_indices]
            print("Top 30 features by variance:")
            for i, (name, variance) in enumerate(zip(top_30_feature_names, top_30_variances)):
                print(f"{i+1}. {name}: {variance:.2f}")
            return df[top_30_feature_names]
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in select_top_30_features_with_highest_variance method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def handle_outliers_using_winsorization(self, train_df):
        self.logger_object.log(
            self.file_object, "Entered the handle_outliers_using_winsorization method in Preprocessor class.")
        try:
            # Calculate the 5th and 95th percentiles for each column
            percentiles = train_df.quantile([0.01, 0.99])

            # Get columns where max value is greater than 95th percentile or min value is less than 5th percentile
            cols_to_clip = train_df.columns[(train_df.max() > percentiles.loc[0.99]).values | (
                train_df.min() < percentiles.loc[0.01]).values]

            # Clip values between the 5th and 95th percentile for selected columns
            winsorized = train_df.copy()
            winsorized[cols_to_clip] = np.clip(
                train_df[cols_to_clip], a_min=percentiles.loc[0.01, cols_to_clip], a_max=percentiles.loc[0.99, cols_to_clip], axis=1)

            # Print columns where values were clipped
            print('Columns where values were clipped:')
            print(len(cols_to_clip))
            print(cols_to_clip)

            # Print quantile range for each column
            new_dfs = []
            for feature in train_df.columns:
                # find the minimum and maximum value for that feature
                feature_min = train_df[feature].quantile(0.01)
                feature_max = train_df[feature].quantile(0.99)
                new_df = pd.DataFrame({
                    'Feature': [feature],
                    'Min Value': [feature_min],
                    'Max Value': [feature_max]
                })
                new_dfs.append(new_df)
            result_df = pd.concat(new_dfs, ignore_index=True)
            print(result_df)

            # Print the winsorized DataFrame
            print('\nWinsorized DataFrame:')
            print(winsorized)
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in handle_outliers_using_winsorization method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def check_feature_imbalance(self, train_df):
        self.logger_object.log(
            self.file_object, "Entered the check_feature_imbalance method in Preprocessor class.")
        try:
            for feature in train_df.columns:
                class_proportions = train_df[feature].value_counts(
                    normalize=True)
                print(f"Feature: {feature}")
                print(f"Class proportions:")
                print(class_proportions)
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in check_feature_imbalance method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def handle_feature_imbalance(self, train_df):
        self.logger_object.log(
            self.file_object, "Entered the handle_feature_imbalance method in Preprocessor class.")
        try:
            for feature in train_df.columns:
                X = train_df[feature].values.reshape(-1, 1)
                y = train_df['Target'].values
                if len(set(y)) > 1:
                    class_counts = pd.Series(y).value_counts()
                    class_imbalance_ratios = {}
                    for c in set(y):
                        class_imbalance_ratios[c] = class_counts[1] / \
                            class_counts[c]
                    most_imbalanced_class = max(
                        class_imbalance_ratios, key=class_imbalance_ratios.get)
                    if class_imbalance_ratios[most_imbalanced_class] > 2:
                        smote = SMOTE(sampling_strategy={
                            most_imbalanced_class: 'minority'})
                        X_resampled, y_resampled = smote.fit_resample(
                            X, y)
                    elif class_imbalance_ratios[most_imbalanced_class] < 0.5:
                        rus = RandomUnderSampler(
                            sampling_strategy={most_imbalanced_class: 'majority'})
                        X_resampled, y_resampled = rus.fit_resample(X, y)
                    else:
                        X_resampled, y_resampled = X, y
                    train_df[feature] = X_resampled
                    train_df['Target'] = y_resampled
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in handle_feature_imbalance method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def plot_class_distribution(self, train_df):
        self.logger_object.log(
            self.file_object, "Entered the plot_class_distribution method in Preprocessor class.")
        try:
            class_counts = train_df['Target'].value_counts()
            plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%')
            plt.axis('equal')
            plt.show()
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in plot_class_distribution method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def get_top_correlated_features(self, train_df, target_col='Target', num_features=10):
        self.logger_object.log(
            self.file_object, "Entered the get_top_correlated_features method in Preprocessor class.")
        try:
            corr_matrix = train_df.corr()[target_col]
            top_n = corr_matrix.abs().nlargest(num_features)
            return top_n
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in get_top_correlated_features method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def get_relevant_features(self, train_df, target_col='Target', corr_threshold=0.18):
        self.logger_object.log(
            self.file_object, "Entered the get_relevant_features method in Preprocessor class.")
        try:
            corr_matrix = train_df.corr()
            corr_with_target = corr_matrix[target_col].abs()
            relevant_features = corr_with_target[corr_with_target >
                                                 corr_threshold].index.tolist()
            return relevant_features
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in get_relevant_features method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def fit_and_evaluate_models(self, train_df, target_col='Target', test_size=0.2, random_state=42):
        self.logger_object.log(
            self.file_object, "Entered the fit_and_evaluate_models method in Preprocessor class.")
        try:
            X = train_df.drop(target_col, axis=1)
            y = train_df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)
            models = [
                DecisionTreeClassifier(),
                RandomForestClassifier(),
                LogisticRegression()
            ]
            for model in models:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(type(model).__name__, "accuracy:",
                      "{:.2f}".format(accuracy*100))
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in fit_and_evaluate_models method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def train_random_forest_classifier(self, df, test_size=0.2, random_state=42, n_estimators=100):
        self.logger_object.log(
            self.file_object, "Entered the train_random_forest_classifier method in Preprocessor class.")
        try:
            X = df.drop('Target', axis=1)
            y = df['Target']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)

            model = RandomForestClassifier(
                n_estimators=n_estimators, random_state=random_state)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            return accuracy
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in train_random_forest_classifier method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def get_top_features_with_highest_variance(self, train_df, n=30):
        self.logger_object.log(
            self.file_object, "Entered the get_top_features_with_highest_variance method in Preprocessor class.")
        try:
            top_features = train_df.var().sort_values(ascending=False).head(n)
            top_feature_names = top_features.index.tolist()
            return top_feature_names
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in get_top_features_with_highest_variance method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def check_missing_features(self, train_df, top_30_feature_names):
        self.logger_object.log(
            self.file_object, "Entered the check_missing_features method in Preprocessor class.")
        try:
            thresholds = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
                          0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24]
            accuracy_dict = {}

            for threshold in thresholds:
                df = train_df.copy()
                df['Target'] = train_df['Target']
                corr_matrix = df.corr()
                target_variable = 'Target'
                corr_with_target = corr_matrix[target_variable].abs()
                relevant_features = corr_with_target[corr_with_target > threshold].index.tolist(
                )

                for i in range(15):
                    if top_30_feature_names[i] not in relevant_features:
                        relevant_features.append(top_30_feature_names[i])
                        print("threshold:", threshold, "i:", i,
                              "Feature", top_30_feature_names[i])
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in check_missing_features method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def find_top_thresholds(self, train_df, top_30_feature_names):
        self.logger_object.log(
            self.file_object, "Entered the find_top_thresholds method in Preprocessor class.")
        try:
            thresholds = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
                          0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24]
            accuracy_dict = {}
            for threshold in thresholds:
                df = train_df
                df['Target'] = train_df['Target']
                corr_matrix = df.corr()
                target_variable = 'Target'
                corr_with_target = corr_matrix[target_variable].abs()
                relevant_features = corr_with_target[corr_with_target > threshold].index.tolist(
                )
                X = train_df[relevant_features]
                X = X.drop('Target', axis=1)
                y = train_df['Target']
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)
                rf_model = RandomForestClassifier(
                    n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                y_pred = rf_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_dict[threshold] = accuracy
                f1 = f1_score(y_test, y_pred, average='micro')
                print("Threshold: ", threshold, " Accuracy: ", "{:.2f}".format(
                    accuracy_dict[threshold]), " F1 Score: ", "{:.2f}".format(f1))
            sorted_accuracy_dict = sorted(
                accuracy_dict.items(), key=lambda x: x[1], reverse=True)
            print("Top 6 thresholds with the highest accuracy:")
            for i in range(6):
                print("Threshold: ", sorted_accuracy_dict[i][0], " Accuracy: ", "{:.2f}".format(
                    sorted_accuracy_dict[i][1]))
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in find_top_thresholds method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def find_thresholds_accuracy(self, train_df, top_30_feature_names):
        self.logger_object.log(
            self.file_object, "Entered the find_thresholds_accuracy method in Preprocessor class.")
        try:
            thresholds = [0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
            accuracy_dict = {}

            for threshold in thresholds:
                corr_matrix = train_df.corr()
                # Find relevant features based on correlation with target variable
                target_variable = 'Target'
                corr_with_target = corr_matrix[target_variable].abs()
                relevant_features = corr_with_target[corr_with_target > threshold].index.tolist(
                )
                # Train a random forest model on relevant features
                X = train_df[relevant_features].drop('Target', axis=1)
                y = train_df['Target']
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)
                rf_model = RandomForestClassifier(
                    n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                y_pred = rf_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_dict[threshold] = accuracy
                f1 = f1_score(y_test, y_pred, average='micro')
                print(
                    f"Threshold: {threshold:.2f} Accuracy: {accuracy:.2f} F1 Score: {f1:.2f}")
                for i in range(15):
                    if top_30_feature_names[i] not in relevant_features:
                        relevant_features.append(top_30_feature_names[i])
                    X = train_df[relevant_features].drop('Target', axis=1)
                    y = train_df['Target']
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42)
                    rf_model = RandomForestClassifier(
                        n_estimators=100, random_state=42)
                    rf_model.fit(X_train, y_train)
                    y_pred = rf_model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    if accuracy > accuracy_dict[threshold]:
                        accuracy_dict[threshold] = accuracy
                        f1 = f1_score(y_test, y_pred, average='micro')
                        print(
                            f"Threshold: {threshold:.2f} Accuracy: {accuracy:.2f} F1 Score: {f1:.2f}")
                print(
                    f"Number of features at threshold {threshold:.2f}: {len(relevant_features)}")
                print("\n")

            # Find threshold with minimum number of features and accuracy > 0.94
            print()
            print()
            print()
            print("Number of feature for accuracy 96%:")
            print()
            print(X.columns.tolist())
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in find_thresholds_accuracy method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def hyperparameter_tuning(self, train_df, top_30_feature_names):
        self.logger_object.log(
            self.file_object, "Entered the hyperparameter_tuning method in Preprocessor class.")
        try:
            thresholds = [0.16, 0.17]
            accuracy_dict = {}

            # Loop over different correlation thresholds
            for threshold in thresholds:
                # Calculate correlation matrix
                corr_matrix = train_df.corr()
                # Find relevant features based on correlation with target variable
                target_variable = 'Target'
                corr_with_target = corr_matrix[target_variable].abs()
                relevant_features = corr_with_target[corr_with_target > threshold].index.tolist(
                )

                # Train a random forest model on relevant features
                X = train_df[relevant_features].drop('Target', axis=1)
                y = train_df['Target']
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)

                model = RandomForestClassifier(
                    n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                model = RandomForestClassifier(random_state=0)
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, scoring='accuracy')
                grid_search.fit(X_train, y_train)
                print("Best hyperparameters:", grid_search.best_params_)
                y_pred = grid_search.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                print('Accuracy:', accuracy*100)
                f1 = f1_score(y_test, y_pred, average='micro')
                print("F1 score:", f1)
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in hyperparameter_tuning method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def cross_validation(self, train_df):
        self.logger_object.log(
            self.file_object, "Entered the cross_validation method in Preprocessor class.")
        try:
            X = train_df.drop('Target', axis=1)
            y = train_df['Target']

            kfold = KFold(n_splits=5, random_state=42, shuffle=True)

            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            print(cross_val_score(model, X, y, cv=kfold, scoring='accuracy'))

            results = cross_val_score(
                model, X, y, cv=kfold, scoring='accuracy')
            print(results.mean()*100)
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in cross_validation method. Exception "
                                   "message: " + str(e))
            raise Exception()

    def generate_report(self, train_df, top_30_feature_names):
        self.logger_object.log(
            self.file_object, "Entered the generate_report method in Preprocessor class.")
        try:
            accuracy_dict = {}
            threshold = 0.17
            corr_matrix = train_df.corr()
            # Find relevant features based on correlation with target variable
            target_variable = 'Target'
            corr_with_target = corr_matrix[target_variable].abs()
            relevant_features = corr_with_target[corr_with_target > threshold].index.tolist(
            )

            # Train a random forest model on relevant features
            X = train_df[relevant_features].drop('Target', axis=1)
            y = train_df['Target']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            rf_model = RandomForestClassifier(
                n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            best_model = rf_model
            y_pred = rf_model.predict(X_test)
            ytest = y_test
            ypred = y_pred
            accuracy = accuracy_score(ytest, ypred)
            accuracy_dict[threshold] = accuracy

            # Add top 30 features not already included and retrain model
            for i in range(15):
                if top_30_feature_names[i] not in relevant_features:
                    relevant_features.append(top_30_feature_names[i])
                X = train_df[relevant_features].drop('Target', axis=1)
                y = train_df['Target']
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)
                rf_model = RandomForestClassifier(
                    n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                y_pred = rf_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                if accuracy > accuracy_dict[threshold]:
                    accuracy_dict[threshold] = accuracy
                    ytest = y_test
                    ypred = y_pred
                    best_model = rf_model
            # cunfusion matrix
            cm = confusion_matrix(ytest, ypred)
            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(x=j, y=i, s=cm[i, j], va='center',
                            ha='center', size='xx-large')
            plt.xlabel('Predictions', fontsize=18)
            plt.ylabel('Actuals', fontsize=18)
            plt.title('Confusion Matrix', fontsize=18)
            plt.show()
            # Classification Report
            print(classification_report(ytest, ypred))
            return "RandomForest", best_model
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in generate_report method. Exception "
                                   "message: " + str(e))
            raise Exception()

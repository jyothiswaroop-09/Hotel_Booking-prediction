# Import system and dataclass utilities
import sys
from dataclasses import dataclass

# Add project root to the system path to resolve 'src' module imports
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import required libraries
import numpy as np
import pandas as pd

# Scikit-learn modules for preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import custom components
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, plot_qq

# Import SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # Use imblearn's pipeline for SMOTE integration

# Configuration class for storing transformation file path
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

# Main class for data transformation
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for setting up the transformation pipelines
        for both numerical and categorical columns.
        '''
        try:
            numerical_columns = ["lead_time","arrival_date_year","arrival_date_day_of_month",
                                 "stays_in_weekend_nights","adults","total_kids","is_repeated_guest",
                                 "previous_cancellations","agent","days_in_waiting_list",
                                 "adr","required_car_parking_spaces"]
            categorical_columns = [
                "hotel",
                "arrival_date_month",
                "meal",
                "country",
                "market_segment","reserved_room_type","assigned_room_type",
                "deposit_type","customer_type",]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore",sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            
            # Combine children and babies into total_kids
            train_df["total_kids"] = train_df["children"] + train_df["babies"]
            test_df["total_kids"] = test_df["children"] + test_df["babies"]
            
            # # ✅ Check if total_kids was added properly
            # print("Train Columns:", train_df.columns)
            # print("Test Columns:", test_df.columns)
            # print(train_df[["children", "babies", "total_kids"]].head())
            # print(test_df[["children", "babies", "total_kids"]].head())

            cols_to_drop = ["previous_bookings_not_canceled", "company","reservation_status_date","total_of_special_requests"
                            ,"arrival_date_week_number","booking_changes","children","babies","distribution_channel","reservation_status"]
            train_df.drop(columns=cols_to_drop,axis=1, inplace=True, errors='ignore')
            test_df.drop(columns=cols_to_drop,axis=1, inplace=True, errors='ignore')
            
            logging.info(" drop done")
            
             # Handle outliers using IQR method before transformation
            def remove_outliers_iqr(df, numerical_columns):
                for col in numerical_columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                return df
            logging.info("remove outliers") ## added new
            
            numerical_columns = ["lead_time","arrival_date_year","arrival_date_day_of_month","stays_in_weekend_nights","adults",
                                 "total_kids","is_repeated_guest","previous_cancellations","agent","days_in_waiting_list",
                                 "adr","required_car_parking_spaces"]
            # categorical_columns = [
            #     "hotel","arrival_date_year",
            #     "arrival_date_month",
            #     "meal",
            #     "country",
            #     "market_segment","distribution_channel","reserved_room_type","assigned_room_type",
            #     "deposit_type","customer_type","reservation_status"]
            
            
            train_df = remove_outliers_iqr(train_df, numerical_columns)
            test_df = remove_outliers_iqr(test_df, numerical_columns)
            
            # Define the target/output column
            target_column_name = "is_canceled"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Apply the preprocessing steps
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            # ✅ Plot QQ plots for the first few features after transformation
            for i in range(min(3, input_feature_train_arr.shape[1])):  # plot first 3 features
                feature_data = input_feature_train_arr[:, i]
                plot_path = os.path.join("artifacts", f"qqplot_feature_{i}.png")
                plot_qq(
                    feature_data,
                    title=f"QQ Plot - Feature {i}",
                    save_path=plot_path
                )
                logging.info(f"QQ plot saved: {plot_path}") ## added new

            # --- SMOTE Integration ---
            logging.info("Applying SMOTE on the training data.")
            smote = SMOTE(random_state=42)
            # Apply SMOTE to the transformed training features and target
            input_feature_train_resampled, target_feature_train_resampled = smote.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            logging.info(f"Original training data shape: {input_feature_train_arr.shape}, {target_feature_train_df.shape}")
            logging.info(f"Resampled training data shape: {input_feature_train_resampled.shape}, {target_feature_train_resampled.shape}")
            # --- End SMOTE Integration ---

            # Concatenate input features and target values horizontally (column-wise)
            # Use the resampled data for training
            train_arr = np.c_[input_feature_train_resampled, np.array(target_feature_train_resampled)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] # Test data is not resampled

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
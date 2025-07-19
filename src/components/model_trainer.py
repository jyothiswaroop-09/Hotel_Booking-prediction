import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models,plot_qq

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("models")
            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000,n_jobs=-1),
                "XGBClassifier": XGBClassifier(random_state=42,verbosity=0),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False,random_state=42),
                "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
            }
            logging.info("models1")
            param = {
                "Random Forest": {
                'n_estimators': [100],
                'max_depth': [10, 20]
                },

                "Decision Tree": {
                'criterion': ['gini'],
                'max_depth': [10]
                },

                "Gradient Boosting": {
                'n_estimators': [100],
                'learning_rate': [0.1]
                },

                "Logistic Regression": {
                'C': [1.0],
                'solver': ['liblinear'],
                'max_iter': [200]
                },

                "XGBClassifier": {
                'n_estimators': [100],
                'learning_rate': [0.1]
                },

                "CatBoosting Classifier": {
                'depth': [6],
                'learning_rate': [0.1],
                'iterations': [100]
                },

                "AdaBoost Classifier": {
                'n_estimators': [100],
                'learning_rate': [0.1]
                }
            }

            logging.info("models completed")

            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models,param)
            logging.info("models3")
            ## To get best model score from dict
            best_model_score = max(model_report.values())
            logging.info("models4")
            ## To get best model name from dict

            # best_model_name = list(model_report.keys())[
            #     list(model_report.values()).index(best_model_score)
            # ]
            best_model_name  = max(model_report, key=model_report.get)
            
            best_model = models[best_model_name]
            logging.info(f"best_model: {best_model}")

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            final_accuracy_score = accuracy_score(y_test, predicted)
            
            residuals = y_test - predicted # added new 
            qqplot_path = os.path.join("artifacts", "qqplot_residuals.png")
            plot_qq(
                residuals,
                title="QQ Plot - Residuals",
                save_path=qqplot_path
            )
            logging.info(f"Residual QQ plot saved at: {qqplot_path}")

            
            return final_accuracy_score
            



            
        except Exception as e:
            raise CustomException(e,sys)
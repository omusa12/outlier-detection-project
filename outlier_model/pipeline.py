from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from feature_engine.imputation import MeanMedianImputer
from feature_engine.transformation import PowerTransformer

from outlier_model.config.core import config

outlier_pipe = Pipeline([
    ('mean_imputation', MeanMedianImputer(
        imputation_method=config.model_config.imputation_method,
        variables=config.model_config.mean_missing_features,
    )),
    ('power_features', PowerTransformer(
        variables=config.model_config.power_features,
        exp=config.model_config.exp
    )),
    ('scaler', MinMaxScaler()),
    ('XGBoost', XGBClassifier(
        learning_rate=config.model_config.learning_rate,
        n_estimators=config.model_config.n_estimators,
        objective=config.model_config.objective,
        nthread=config.model_config.nthread,
        random_state=config.model_config.random_state,
        subsample=config.model_config.subsample,
        eval_metric=config.model_config.eval_metric,
        min_child_weight=config.model_config.min_child_weight,
        max_depth=config.model_config.max_depth,
        gamma=config.model_config.gamma,
        colsample_bytree=config.model_config.colsample_bytree
    ))
])

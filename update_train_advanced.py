import re
with open('src/aac_adoption/models/train_advanced.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace("metadata = {}", "metadata = {'training_target_min': float(train_y.min()), 'training_target_max': float(train_y.max())}")

replacement_reg = """def train_advanced_regression(
    df: pd.DataFrame,
    models_dir: Path,
    run_timestamp: str,
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    early_stopping_rounds: int,
) -> list[dict[str, Any]]:
    \"\"\"Train CatBoost days-to-outcome regressors with log-transform.\"\"\"
    rows: list[dict[str, Any]] = []
    params = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "early_stopping_rounds": early_stopping_rounds,
        "random_seed": RANDOM_STATE,
    }
    for subset in ANIMAL_SUBSETS:
        split = make_time_split(df, "regression_target_days", animal_subset=subset)
        feature_columns = model_feature_columns(split.train)
        
        split_train = split.train.copy()
        split_train = log_transform_LOS(split_train, "regression_target_days")
        
        split_val = split.validation.copy()
        split_val = log_transform_LOS(split_val, "regression_target_days")
        
        split_test = split.test.copy()
        split_test = log_transform_LOS(split_test, "regression_target_days")
        
        model, metadata = _fit_and_save(
            model=CatBoostRegressor(**params),
            task="regression",
            split=DatasetSplit(
                full_data=split.full_data,
                train=split_train,
                validation=split_val,
                test=split_test,
                strategy=split.strategy,
                train_period=split.train_period,
                validation_period=split.validation_period,
                test_period=split.test_period,
                animal_subset=split.animal_subset,
                selection=split.selection,
            ),
            feature_columns=feature_columns,
            target_column="log_regression_target_days",
            models_dir=models_dir,
            run_timestamp=run_timestamp,
            params=params,
            winsorize_target=True,
        )
        if not split.selection.empty:
            sel_x = prepare_catboost_frame(split.selection, feature_columns)
            sel_predictions = np.exp(model.predict(sel_x)) - 1
            sel_metrics = regression_metrics(split.selection["regression_target_days"], sel_predictions)
            rows.append({
                **metadata,
                **sel_metrics,
                "target_column": "regression_target_days",
                "target_transform": "log1p",
                "prediction_inverse_transform": "expm1",
                "metric_split": "selection",
                "selection_eligible": 1,
            })
            
        if not split.test.empty:
            test_x = prepare_catboost_frame(split.test, feature_columns)
            predictions = np.exp(model.predict(test_x)) - 1
            metrics = regression_metrics(split.test["regression_target_days"], predictions)
            rows.append({
                **metadata,
                **metrics,
                "target_column": "regression_target_days",
                "target_transform": "log1p",
                "prediction_inverse_transform": "expm1",
                "metric_split": "test",
                "selection_eligible": 0,
            })
    return rows"""
text = re.sub(r'def train_advanced_regression\([\s\S]*?return rows', replacement_reg, text, count=1)

with open('src/aac_adoption/models/train_advanced.py', 'w', encoding='utf-8') as f:
    f.write(text)

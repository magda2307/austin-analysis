import re
with open('src/aac_adoption/models/train_adopted_regression.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('metadata["target_transform"] = "log1p"', 'metadata["target_transform"] = "log1p"\n    metadata["training_target_min"] = float(train_y.min())\n    metadata["training_target_max"] = float(train_y.max())')

replacement_func = """def train_adopted_regression(
    df: pd.DataFrame,
    models_dir: Path,
    run_timestamp: str,
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    early_stopping_rounds: int,
) -> list[dict[str, Any]]:
    \"\"\"Train CatBoost days-to-adoption regressors on adopted subset.\"\"\"
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
    
    # Filter to adopted only
    adopted_df = df.loc[
        df["classification_target"].eq(1) & df["days_to_adoption"].notna()
    ].copy()

    for subset in ANIMAL_SUBSETS:
        split = make_time_split(adopted_df, "days_to_adoption", animal_subset=subset)
        feature_columns = model_feature_columns(split.train)
        model, metadata = _fit_and_save_adopted(
            model=CatBoostRegressor(**params),
            task="regression_adopted",
            split=split,
            feature_columns=feature_columns,
            target_column="days_to_adoption",
            models_dir=models_dir,
            run_timestamp=run_timestamp,
            params=params,
        )
        
        if not split.selection.empty:
            sel_x = prepare_catboost_frame(split.selection, feature_columns)
            log_predictions = model.predict(sel_x)
            predictions = np.expm1(log_predictions)
            predictions = np.maximum(predictions, 0.0)
            sel_metrics = regression_metrics(split.selection["days_to_adoption"], predictions)
            rows.append({
                **metadata,
                **sel_metrics,
                "target_column": "days_to_adoption",
                "target_transform": "log1p",
                "prediction_inverse_transform": "expm1",
                "metric_split": "selection",
                "selection_eligible": 1,
            })
            
        if not split.test.empty:
            test_x = prepare_catboost_frame(split.test, feature_columns)
            log_predictions = model.predict(test_x)
            predictions = np.expm1(log_predictions)
            predictions = np.maximum(predictions, 0.0)
            metrics = regression_metrics(split.test["days_to_adoption"], predictions)
            rows.append({
                **metadata,
                **metrics,
                "target_column": "days_to_adoption",
                "target_transform": "log1p",
                "prediction_inverse_transform": "expm1",
                "metric_split": "test",
                "selection_eligible": 0,
            })
    return rows"""
text = re.sub(r'def train_adopted_regression\([\s\S]*?return rows', replacement_func, text, count=1)

with open('src/aac_adoption/models/train_adopted_regression.py', 'w', encoding='utf-8') as f:
    f.write(text)

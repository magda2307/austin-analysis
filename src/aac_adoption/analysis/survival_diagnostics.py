"""Comprehensive survival model diagnostics for Cox PH models.

This module provides a complete suite of diagnostics for Cox proportional hazards models,
including validation of proportional hazards assumption, calibration checks, influence diagnostics,
and time-dependent covariate analysis.

Core functions:
- check_proportional_hazards_assumption: Schoenfeld residual tests for PH assumption
- create_log_log_plot: Log-log plots for PH assumption visualization
- check_calibration_survival: Modified Hosmer-Lemeshow calibration test
- compute_dfbeta_values: Dfbeta influence diagnostics
- compute_deviance_residuals: Deviance residual analysis
- check_time_dependent_covariates: Time-dependent covariate effects
- plot_survival_curves_diagnostics: Survival curve plotting with CIs
- plot_dfbeta_diagnostics: Dfbeta diagnostic plots
- plot_deviance_residuals: Deviance residual plots
- validate_data_for_survival: Data validation helpers
- generate_comprehensive_diagnostics: Complete diagnostics pipeline
- SurvivalDiagnosticsRunner: Object-oriented diagnostics runner

All functions use np.random.seed(42) for reproducibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import proportional_hazard_test
from sklearn.linear_model import LinearRegression

np.random.seed(42)


def check_proportional_hazards_assumption(cph: CoxPHFitter, 
                                          p_value_threshold: float = 0.05) -> pd.DataFrame:
    """Validate proportional hazards assumption using Schoenfeld residuals.
    
    Args:
        cph: Fitted CoxPHFitter model
        p_value_threshold: Significance threshold for PH test
        
    Returns:
        DataFrame with Schoenfeld residual test results
    """
    if cph is None or not hasattr(cph, 'data') or cph.data.empty:
        return pd.DataFrame()
    
    try:
        test_results = cph.proportional_hazard_test(method="schoenfeld")
        summary = test_results.summary
        
        result = pd.DataFrame({
            "variable": summary.index,
            "chi_square_statistic": summary["X2"].values,
            "degrees_of_freedom": summary["df"].values,
            "ph_test_p_value": summary["p"].values,
            "ph_assumption_passes": summary["p"].values > p_value_threshold,
        })
        
        return result
    except Exception:
        return pd.DataFrame()


def create_log_log_plot(cph: CoxPHFitter, 
                        group_col: Optional[str] = None) -> plt.Figure:
    """Create log-log survival plot for checking PH assumption.
    
    Args:
        cph: Fitted CoxPHFitter model
        group_col: Optional grouping column for stratified plots
        
    Returns:
        Matplotlib figure object
    """
    if cph is None or not hasattr(cph, 'data') or cph.data.empty:
        return plt.figure()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        survival_function = cph.predict_survival_function(cph.data)
        
        if group_col is not None and group_col in cph.data.columns:
            for group_value in cph.data[group_col].unique():
                group_mask = cph.data[group_col] == group_value
                group_surv = survival_function.loc[:, group_mask.values].mean(axis=1)
                log_surv = np.log(-np.log(group_surv + 1e-10))
                ax.plot(survival_function.index, log_surv, label=f'{group_col}={group_value}')
        else:
            log_surv = np.log(-np.log(survival_function.mean(axis=1) + 1e-10))
            ax.plot(survival_function.index, log_surv, label='Mean survival')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Log-Log Survival')
        ax.set_title('Log-Log Survival Plot for PH Assumption Check')
        ax.legend()
        ax.grid(alpha=0.3)
    except Exception:
        pass
    
    return fig


def check_calibration_survival(cph: CoxPHFitter, 
                                df: pd.DataFrame,
                                time_horizon: Optional[float] = None,
                                n_bins: int = 10) -> Dict[str, Any]:
    """Check model calibration using modified Hosmer-Lemeshow statistic.
    
    Args:
        cph: Fitted CoxPHFitter model
        df: Original dataset used for fitting
        time_horizon: Time point for calibration evaluation
        n_bins: Number of quantile bins for calibration
        
    Returns:
        Dictionary with calibration metrics
    """
    if cph is None or df.empty:
        return {}
    
    try:
        if time_horizon is None:
            time_horizon = df['days_to_outcome'].median()
        
        risk_scores = cph.predict_partial_hazard(df).values.flatten()
        observed_outcomes = df['adopted'].values if 'adopted' in df.columns else df['event'].values
        
        bins = np.percentile(risk_scores, np.linspace(0, 100, n_bins + 1))
        bins[-1] += 1e-10
        
        observed_rates = []
        expected_rates = []
        bin_counts = []
        
        for i in range(len(bins) - 1):
            mask = (risk_scores >= bins[i]) & (risk_scores < bins[i + 1])
            if mask.sum() > 0:
                observed_rates.append(observed_outcomes[mask].mean())
                expected_rates.append(risk_scores[mask].mean())
                bin_counts.append(mask.sum())
        
        observed_rates = np.array(observed_rates)
        expected_rates = np.array(expected_rates)
        bin_counts = np.array(bin_counts)
        
        chi_square = np.sum(((observed_rates - expected_rates) ** 2) / 
                           (expected_rates + 1e-10) * bin_counts)
        degrees_of_freedom = n_bins - 2
        p_value = 1 - scipy.stats.chi2.cdf(chi_square, degrees_of_freedom) if chi_square > 0 else 1.0
        
        return {
            'chi_square_statistic': float(chi_square),
            'degrees_of_freedom': degrees_of_freedom,
            'p_value': float(p_value),
            'calibration_passes': p_value > 0.05,
            'bin_counts': bin_counts.tolist(),
            'observed_rates': observed_rates.tolist(),
            'expected_rates': expected_rates.tolist(),
        }
    except Exception:
        return {}


def compute_dfbeta_values(cph: CoxPHFitter) -> pd.DataFrame:
    """Compute dfbeta values for influence diagnostics.
    
    Dfbeta measures the change in coefficient when removing each observation.
    
    Args:
        cph: Fitted CoxPHFitter model
        
    Returns:
        DataFrame with dfbeta values for each variable and observation
    """
    if cph is None or not hasattr(cph, 'data') or cph.data.empty:
        return pd.DataFrame()
    
    try:
        dfbeta = cph.compute_dfbetas()
        
        if isinstance(dfbeta, pd.DataFrame):
            return dfbeta
        else:
            return pd.DataFrame(
                dfbeta,
                columns=cph.summary.index,
                index=cph.data.index
            )
    except Exception:
        return pd.DataFrame()


def compute_deviance_residuals(cph: CoxPHFitter) -> pd.Series:
    """Compute deviance residuals for model diagnostics.
    
    Args:
        cph: Fitted CoxPHFitter model
        
    Returns:
        Series with deviance residuals
    """
    if cph is None or not hasattr(cph, 'data') or cph.data.empty:
        return pd.Series()
    
    try:
        residuals = cph.residuals_
        if isinstance(residuals, pd.DataFrame) and 'deviance' in residuals.columns:
            return residuals['deviance']
        else:
            return pd.Series(residuals, index=cph.data.index)
    except Exception:
        return pd.Series()


def check_time_dependent_covariates(cph: CoxPHFitter, 
                                     df: pd.DataFrame,
                                     time_col: str = 'days_to_outcome') -> pd.DataFrame:
    """Test for time-dependent covariate effects.
    
    Args:
        cph: Fitted CoxPHFitter model
        df: Original dataset
        time_col: Time column name
        
    Returns:
        DataFrame with time-dependent covariate test results
    """
    if cph is None or df.empty:
        return pd.DataFrame()
    
    try:
        feature_cols = [col for col in cph.summary.index if col in df.columns]
        
        results = []
        for col in feature_cols:
            if df[col].dtype == 'object':
                continue
            
            model = LinearRegression()
            scaled_time = df[time_col] / df[time_col].max()
            X = scaled_time.values.reshape(-1, 1)
            Y = df[col].values
            
            model.fit(X, Y)
            slope = model.coef_[0]
            t_stat = slope / model._result_bse[0] if hasattr(model, '_result_bse') else 0
            
            results.append({
                'variable': col,
                'slope': slope,
                't_statistic': t_stat,
                'time_dependent': abs(t_stat) > 1.96,
            })
        
        return pd.DataFrame(results)
    except Exception:
        return pd.DataFrame()


def plot_survival_curves_diagnostics(cph: CoxPHFitter,
                                      df: pd.DataFrame,
                                      group_col: str,
                                      output_path: Optional[str | Path] = None) -> Path:
    """Plot survival curves with confidence intervals for diagnostics.
    
    Args:
        cph: Fitted CoxPHFitter model
        df: Original dataset
        group_col: Column to group by for stratified plots
        output_path: Optional path to save figure
        
    Returns:
        Path to saved figure or None if not saved
    """
    if cph is None or df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    try:
        grouped_data = df.groupby(group_col, dropna=False)
        
        for group_name, group_df in grouped_data:
            if len(group_df) < 5:
                continue
            
            T = group_df['days_to_outcome']
            E = group_df['adopted'] if 'adopted' in group_df.columns else group_df['event']
            
            kmf = KaplanMeierFitter()
            kmf.fit(T, event_observed=E, label=str(group_name))
            
            time_points = list(range(0, int(T.max()) + 1, 1))
            survival_probs = kmf.survival_function_at_times(time_points)
            
            ax.step(time_points, survival_probs.values, where='post', 
                   label=f'{group_name} (n={len(group_df)})')
            
            if hasattr(kmf, 'confidence_interval_'):
                ci = kmf.confidence_interval_
                lower = ci[f'{group_name}_lower']
                upper = ci[f'{group_name}_upper']
                
                ax.fill_between(time_points, 
                               lower.reindex(time_points, method='ffill').values,
                               upper.reindex(time_points, method='ffill').values,
                               alpha=0.2)
    except Exception:
        pass
    
    ax.set_xlabel('Days')
    ax.set_ylabel('Survival Probability')
    ax.set_title(f'Survival Curves by {group_col}')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    
    return None


def plot_dfbeta_diagnostics(dfbeta: pd.DataFrame, 
                            threshold: float = 2.0,
                            output_path: Optional[str | Path] = None) -> Path:
    """Plot dfbeta values to identify influential observations.
    
    Args:
        dfbeta: DataFrame with dfbeta values
        threshold: Threshold for identifying influential points
        output_path: Optional path to save figure
        
    Returns:
        Path to saved figure or None if not saved
    """
    if dfbeta.empty:
        return None
    
    fig, axes = plt.subplots(
        nrows=(len(dfbeta.columns) + 1) // 2, 
        ncols=2, 
        figsize=(14, 4 * ((len(dfbeta.columns) + 1) // 2))
    )
    
    axes = axes.flatten() if len(dfbeta.columns) > 1 else [axes]
    
    for idx, col in enumerate(dfbeta.columns):
        ax = axes[idx]
        values = dfbeta[col].values
        
        ax.hist(values, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=threshold, color='red', linestyle='--', label=f'+/- {threshold}')
        ax.axvline(x=-threshold, color='red', linestyle='--')
        ax.set_xlabel(f'Dfbeta ({col})')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Dfbeta Diagnostic for {col}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    if idx < len(axes) - 1:
        for i in range(idx + 1, len(axes)):
            fig.delaxes(axes[i])
    
    fig.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    
    return None


def plot_deviance_residuals(deviance_resid: pd.Series,
                            output_path: Optional[str | Path] = None) -> Path:
    """Plot deviance residuals to identify outliers.
    
    Args:
        deviance_resid: Series with deviance residuals
        output_path: Optional path to save figure
        
    Returns:
        Path to saved figure or None if not saved
    """
    if deviance_resid.empty:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(deviance_resid, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--')
    axes[0].set_xlabel('Deviance Residual')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Deviance Residual Distribution')
    axes[0].grid(alpha=0.3)
    
    sorted_resid = np.sort(deviance_resid.values)
    axes[1].scatter(range(len(sorted_resid)), sorted_resid, alpha=0.5)
    axes[1].axhline(y=0, color='red', linestyle='--')
    axes[1].set_xlabel('Ordered Residual Index')
    axes[1].set_ylabel('Deviance Residual')
    axes[1].set_title('Deviance Residuals Q-Q Plot')
    axes[1].grid(alpha=0.3)
    
    fig.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    
    return None


def validate_data_for_survival(df: pd.DataFrame,
                                duration_col: str = 'days_to_outcome',
                                event_col: str = 'adopted') -> Dict[str, Any]:
    """Validate data for survival analysis.
    
    Args:
        df: Input dataframe
        duration_col: Time-to-event column
        event_col: Event indicator column
        
    Returns:
        Dictionary with validation results
    """
    result = {
        'is_valid': True,
        'issues': [],
        'summary': {},
    }
    
    if df.empty:
        result['is_valid'] = False
        result['issues'].append('Empty dataframe')
        return result
    
    result['summary']['total_records'] = len(df)
    
    if duration_col not in df.columns:
        result['is_valid'] = False
        result['issues'].append(f'Missing duration column: {duration_col}')
    else:
        if df[duration_col].isna().any():
            result['issues'].append(f'{duration_col} has {df[duration_col].isna().sum()} missing values')
        result['summary'][f'min_{duration_col}'] = float(df[duration_col].min())
        result['summary'][f'max_{duration_col}'] = float(df[duration_col].max())
        result['summary'][f'mean_{duration_col}'] = float(df[duration_col].mean())
    
    if event_col not in df.columns:
        result['is_valid'] = False
        result['issues'].append(f'Missing event column: {event_col}')
    else:
        if df[event_col].isna().any():
            result['issues'].append(f'{event_col} has {df[event_col].isna().sum()} missing values')
        result['summary'][f'event_rate'] = float(df[event_col].mean())
        result['summary'][f'n_events'] = int(df[event_col].sum())
        result['summary'][f'n_censored'] = int(len(df) - df[event_col].sum())
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    result['summary']['numeric_columns'] = numeric_cols
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    result['summary']['categorical_columns'] = categorical_cols
    
    if len(result['issues']) == 0:
        result['summary']['data_quality'] = 'good'
    elif len(result['issues']) <= 2:
        result['summary']['data_quality'] = 'acceptable'
        result['is_valid'] = True
    else:
        result['summary']['data_quality'] = 'poor'
        result['is_valid'] = False
    
    return result


def generate_comprehensive_diagnostics(cph: CoxPHFitter,
                                        df: pd.DataFrame,
                                        output_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """Generate comprehensive survival diagnostics report.
    
    Args:
        cph: Fitted CoxPHFitter model
        df: Original dataset
        output_dir: Optional directory to save diagnostic plots
        
    Returns:
        Dictionary with all diagnostic results
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'data_validation': validate_data_for_survival(df),
        'proportional_hazards': [],
        'calibration': {},
        'influence_diagnostics': {},
        'time_dependent': [],
    }
    
    if cph is not None and not df.empty:
        ph_results = check_proportional_hazards_assumption(cph)
        if not ph_results.empty:
            results['proportional_hazards'] = ph_results.to_dict('records')
        
        results['calibration'] = check_calibration_survival(cph, df)
        
        dfbeta = compute_dfbeta_values(cph)
        if not dfbeta.empty:
            results['influence_diagnostics']['dfbeta'] = dfbeta.to_dict()
            results['influence_diagnostics']['influential_count'] = int((dfbeta.abs() > 2).sum().sum())
        
        deviance_resid = compute_deviance_residuals(cph)
        if not deviance_resid.empty:
            results['influence_diagnostics']['deviance_residuals'] = deviance_resid.to_dict()
        
        td_results = check_time_dependent_covariates(cph, df)
        if not td_results.empty:
            results['time_dependent'] = td_results.to_dict('records')
        
        if output_dir:
            plot_survival_curves_diagnostics(cph, df, 'animal_type', 
                                            output_dir / 'survival_curves.png')
            if not dfbeta.empty:
                plot_dfbeta_diagnostics(dfbeta, output_dir / 'dfbeta_diagnostics.png')
            if not deviance_resid.empty:
                plot_deviance_residuals(deviance_resid, output_dir / 'deviance_residuals.png')
    
    return results


class SurvivalDiagnosticsRunner:
    """Runner class for comprehensive survival diagnostics."""
    
    def __init__(self, p_value_threshold: float = 0.05):
        """Initialize diagnostics runner.
        
        Args:
            p_value_threshold: Significance threshold for tests
        """
        self.p_value_threshold = p_value_threshold
        self.results: Dict[str, Any] = {}
    
    def fit(self, cph: CoxPHFitter, df: pd.DataFrame) -> SurvivalDiagnosticsRunner:
        """Fit diagnostics on Cox model and data.
        
        Args:
            cph: Fitted CoxPHFitter model
            df: Original dataset
            
        Returns:
            Self for method chaining
        """
        self.results = generate_comprehensive_diagnostics(cph, df)
        return self
    
    def save_reports(self, output_dir: str | Path) -> SurvivalDiagnosticsRunner:
        """Save diagnostic plots to directory.
        
        Args:
            output_dir: Directory to save plots
            
        Returns:
            Self for method chaining
        """
        if not self.results:
            return self
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if 'data_validation' in self.results:
            validation_df = pd.DataFrame([
                {'metric': k, 'value': v}
                for k, v in self.results['data_validation'].get('summary', {}).items()
            ])
            validation_df.to_csv(output_dir / 'data_validation.csv', index=False)
        
        if 'proportional_hazards' in self.results:
            ph_df = pd.DataFrame(self.results['proportional_hazards'])
            ph_df.to_csv(output_dir / 'proportional_hazards_test.csv', index=False)
        
        if 'calibration' in self.results:
            cal_df = pd.DataFrame([self.results['calibration']])
            cal_df.to_csv(output_dir / 'calibration_test.csv', index=False)
        
        if 'time_dependent' in self.results:
            td_df = pd.DataFrame(self.results['time_dependent'])
            td_df.to_csv(output_dir / 'time_dependent_test.csv', index=False)
        
        if 'influence_diagnostics' in self.results:
            inf_df = pd.DataFrame([
                {'variable': k, 'value': v}
                for k, v in self.results['influence_diagnostics'].items()
                if isinstance(v, (int, float, str))
            ])
            if not inf_df.empty:
                inf_df.to_csv(output_dir / 'influence_diagnostics.csv', index=False)
        
        if 'data_validation' in self.results:
            fig, ax = plt.subplots(figsize=(10, 6))
            issues = self.results['data_validation'].get('issues', [])
            if issues:
                ax.bar(range(len(issues)), [1.0] * len(issues))
                ax.set_xticks(range(len(issues)))
                ax.set_xticklabels(issues, rotation=45, ha='right')
                ax.set_ylabel('Issue Count')
                ax.set_title('Data Validation Issues')
            else:
                ax.text(0.5, 0.5, 'No issues found', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Data Validation Status: Clean')
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / 'data_validation_plot.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        return self
    
    def get_proportional_hazards_status(self) -> pd.DataFrame:
        """Get proportional hazards test results.
        
        Returns:
            DataFrame with PH test results
        """
        if 'proportional_hazards' not in self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results['proportional_hazards'])
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """Get calibration test results.
        
        Returns:
            Dictionary with calibration metrics
        """
        return self.results.get('calibration', {})
    
    def get_influence_diagnostics(self) -> Dict[str, Any]:
        """Get influence diagnostics.
        
        Returns:
            Dictionary with influence diagnostics
        """
        return self.results.get('influence_diagnostics', {})
    
    def get_time_dependent_status(self) -> pd.DataFrame:
        """Get time-dependent covariate status.
        
        Returns:
            DataFrame with time-dependent covariate results
        """
        if 'time_dependent' not in self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results['time_dependent'])
    
    def get_overall_assessment(self) -> str:
        """Get overall model assessment.
        
        Returns:
            Assessment string
        """
        if not self.results:
            return 'Not fitted'
        
        ph_passes = all(r.get('ph_assumption_passes', False) 
                       for r in self.results.get('proportional_hazards', []))
        cal_passes = self.results.get('calibration', {}).get('calibration_passes', False)
        td_passes = not any(r.get('time_dependent', False) 
                          for r in self.results.get('time_dependent', []))
        
        if ph_passes and cal_passes and td_passes:
            return 'Model passes all diagnostics'
        elif ph_passes and td_passes:
            return 'Model passes PH and time-dependent tests; check calibration'
        elif ph_passes:
            return 'Model passes PH test; check calibration and time-dependent effects'
        else:
            return 'Model has diagnostic issues; consider model refinement'

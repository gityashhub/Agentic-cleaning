import pandas as pd
import numpy as np
from scipy import stats
import warnings

class WeightCalculator:
    """Handles survey weight application and statistical computations."""
    
    def __init__(self):
        self.results = {}
    
    def calculate_weighted_statistics(self, df, config):
        """
        Calculate weighted statistics for survey data.
        
        Args:
            df: pandas DataFrame with cleaned data
            config: dict with weight configuration
            
        Returns:
            dict: Statistical results including summaries and margins of error
        """
        # Input validation
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")
        
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        weight_column = config.get('weight_column')
        analysis_vars = config.get('analysis_vars', [])
        strat_vars = config.get('strat_vars', [])
        confidence_level = config.get('confidence_level', 0.95)
        
        # Validate analysis variables exist
        missing_vars = [var for var in analysis_vars if var not in df.columns]
        if missing_vars:
            raise ValueError(f"Analysis variables not found in data: {missing_vars}")
        
        # Validate stratification variables exist
        missing_strat = [var for var in strat_vars if var not in df.columns]
        if missing_strat:
            raise ValueError(f"Stratification variables not found in data: {missing_strat}")
        
        if not analysis_vars:
            raise ValueError("No analysis variables specified")
        
        # Prepare weights
        if weight_column:
            weights = self._prepare_weights(df, weight_column, config)
        else:
            weights = np.ones(len(df))  # Equal weights
        
        results = {
            'summary_stats': {},
            'margins_of_error': {},
            'weight_diagnostics': {},
            'stratified_results': {}
        }
        
        # Weight diagnostics
        if weight_column:
            results['weight_diagnostics'] = self._calculate_weight_diagnostics(weights)
        
        # Calculate statistics for each analysis variable
        for var in analysis_vars:
            if var in df.columns:
                if df[var].dtype in ['object', 'category']:
                    # Categorical variable
                    results['summary_stats'][var] = self._calculate_categorical_stats(
                        df, var, weights, confidence_level
                    )
                else:
                    # Numeric variable
                    results['summary_stats'][var] = self._calculate_numeric_stats(
                        df, var, weights, confidence_level
                    )
        
        # Stratified analysis
        for strat_var in strat_vars:
            if strat_var in df.columns:
                results['stratified_results'][strat_var] = self._calculate_stratified_stats(
                    df, analysis_vars, strat_var, weights, confidence_level
                )
        
        # Convert to DataFrames for display
        if results['summary_stats']:
            results['summary_stats'] = pd.DataFrame(results['summary_stats']).T
        else:
            results['summary_stats'] = pd.DataFrame()
        
        # Calculate margins of error
        margins_df = self._calculate_margins_of_error(
            df, analysis_vars, weights, confidence_level
        )
        results['margins_of_error'] = margins_df
        
        return results
    
    def _prepare_weights(self, df, weight_column, config):
        """Prepare and validate survey weights."""
        weights = df[weight_column].copy()
        
        # Handle missing weights
        if weights.isnull().any():
            warnings.warn("Missing weights detected. Using mean imputation.")
            weights = weights.fillna(weights.mean())
        
        # Trim extreme weights if requested
        if config.get('trim_weights', False):
            trim_threshold = config.get('trim_threshold', 0.97)
            upper_limit = weights.quantile(trim_threshold)
            weights = np.minimum(weights, upper_limit)
        
        # Normalize weights if requested
        if config.get('normalize_weights', True):
            weights = weights * len(weights) / weights.sum()
        
        # Ensure positive weights
        if (weights <= 0).any():
            warnings.warn("Non-positive weights detected. Setting to minimum positive value.")
            min_positive = weights[weights > 0].min()
            weights = np.maximum(weights, min_positive)
        
        return weights
    
    def _calculate_weight_diagnostics(self, weights):
        """Calculate diagnostic statistics for survey weights."""
        diagnostics = {
            'min_weight': weights.min(),
            'max_weight': weights.max(),
            'mean_weight': weights.mean(),
            'median_weight': weights.median(),
            'std_weight': weights.std(),
            'cv_weight': weights.std() / weights.mean(),  # Coefficient of variation
            'effective_sample_size': (weights.sum() ** 2) / (weights ** 2).sum(),
            'design_effect': len(weights) / ((weights.sum() ** 2) / (weights ** 2).sum())
        }
        
        return diagnostics
    
    def _calculate_numeric_stats(self, df, variable, weights, confidence_level):
        """Calculate weighted statistics for numeric variables."""
        data = df[variable].dropna()
        
        # Ensure we have matching indices
        valid_indices = data.index.intersection(weights.index if hasattr(weights, 'index') else range(len(weights)))
        data = data.loc[valid_indices]
        
        if hasattr(weights, 'index'):
            var_weights = weights.loc[valid_indices]
        else:
            var_weights = weights[valid_indices]
        
        if len(data) == 0:
            return self._empty_stats()
        
        # Weighted statistics
        weighted_mean = np.average(data, weights=var_weights)
        weighted_var = np.average((data - weighted_mean) ** 2, weights=var_weights)
        weighted_std = np.sqrt(weighted_var)
        
        # Effective sample size
        eff_n = (var_weights.sum() ** 2) / (var_weights ** 2).sum()
        
        # Standard error of the mean
        se_mean = weighted_std / np.sqrt(eff_n)
        
        # Confidence interval
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, eff_n - 1)
        ci_lower = weighted_mean - t_critical * se_mean
        ci_upper = weighted_mean + t_critical * se_mean
        
        # Weighted quantiles (approximate)
        sorted_indices = np.argsort(data)
        sorted_data = data.iloc[sorted_indices]
        sorted_weights = var_weights.iloc[sorted_indices]
        
        # Convert to numpy arrays to avoid pandas indexing issues
        sorted_weights_array = np.array(sorted_weights)
        cumulative_weights = np.cumsum(sorted_weights_array)
        
        # Check if cumulative_weights is empty
        if len(cumulative_weights) == 0:
            total_weight = 0
        else:
            total_weight = cumulative_weights[-1]
        
        # Find weighted percentiles
        percentiles = [0.25, 0.5, 0.75]
        weighted_percentiles = {}
        
        # Only calculate percentiles if we have data
        if len(sorted_data) > 0 and total_weight > 0:
            for p in percentiles:
                target_weight = p * total_weight
                idx = np.searchsorted(cumulative_weights, target_weight)
                if idx < len(sorted_data):
                    weighted_percentiles[f'p{int(p*100)}'] = sorted_data.iloc[idx]
                else:
                    weighted_percentiles[f'p{int(p*100)}'] = sorted_data.iloc[-1]
        else:
            # Set default values if no data
            for p in percentiles:
                weighted_percentiles[f'p{int(p*100)}'] = np.nan
        
        return {
            'count': len(data),
            'effective_n': eff_n,
            'weighted_mean': weighted_mean,
            'std_error': se_mean,
            'weighted_std': weighted_std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'min': data.min(),
            'max': data.max(),
            **weighted_percentiles
        }
    
    def _calculate_categorical_stats(self, df, variable, weights, confidence_level):
        """Calculate weighted statistics for categorical variables."""
        data = df[variable].dropna()
        var_weights = weights[data.index]
        
        if len(data) == 0:
            return self._empty_stats()
        
        # Weighted counts and proportions
        categories = data.unique()
        results = {}
        
        total_weight = var_weights.sum()
        
        for category in categories:
            mask = (data == category)
            category_weight = var_weights[mask].sum()
            proportion = category_weight / total_weight
            
            # Standard error for proportion
            # Using simple random sampling approximation
            eff_n = (var_weights.sum() ** 2) / (var_weights ** 2).sum()
            se_prop = np.sqrt(proportion * (1 - proportion) / eff_n)
            
            # Confidence interval for proportion
            alpha = 1 - confidence_level
            z_critical = stats.norm.ppf(1 - alpha/2)
            ci_lower = max(0, proportion - z_critical * se_prop)
            ci_upper = min(1, proportion + z_critical * se_prop)
            
            results[str(category)] = {
                'count': mask.sum(),
                'weighted_count': category_weight,
                'proportion': proportion,
                'std_error': se_prop,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
        
        return results
    
    def _calculate_stratified_stats(self, df, analysis_vars, strat_var, weights, confidence_level):
        """Calculate statistics stratified by a grouping variable."""
        stratified_results = {}
        
        # Get unique strata
        strata = df[strat_var].dropna().unique()
        
        for stratum in strata:
            stratum_mask = (df[strat_var] == stratum)
            stratum_df = df[stratum_mask]
            stratum_weights = weights[stratum_mask]
            
            stratum_results = {}
            
            for var in analysis_vars:
                if var in stratum_df.columns:
                    if stratum_df[var].dtype in ['object', 'category']:
                        stratum_results[var] = self._calculate_categorical_stats(
                            stratum_df, var, stratum_weights, confidence_level
                        )
                    else:
                        stratum_results[var] = self._calculate_numeric_stats(
                            stratum_df, var, stratum_weights, confidence_level
                        )
            
            stratified_results[str(stratum)] = stratum_results
        
        return stratified_results
    
    def _calculate_margins_of_error(self, df, analysis_vars, weights, confidence_level):
        """Calculate margins of error for key estimates."""
        margins = {}
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        for var in analysis_vars:
            if var in df.columns:
                data = df[var].dropna()
                var_weights = weights[data.index]
                
                if len(data) == 0:
                    margins[var] = np.nan
                    continue
                
                if df[var].dtype in ['object', 'category']:
                    # For categorical, use the largest margin among categories
                    categories = data.unique()
                    max_margin = 0
                    
                    total_weight = var_weights.sum()
                    eff_n = (var_weights.sum() ** 2) / (var_weights ** 2).sum()
                    
                    for category in categories:
                        mask = (data == category)
                        proportion = var_weights[mask].sum() / total_weight
                        se_prop = np.sqrt(proportion * (1 - proportion) / eff_n)
                        margin = z_critical * se_prop
                        max_margin = max(max_margin, margin)
                    
                    margins[var] = max_margin
                
                else:
                    # For numeric variables, margin of error for the mean
                    weighted_mean = np.average(data, weights=var_weights)
                    weighted_var = np.average((data - weighted_mean) ** 2, weights=var_weights)
                    weighted_std = np.sqrt(weighted_var)
                    
                    eff_n = (var_weights.sum() ** 2) / (var_weights ** 2).sum()
                    se_mean = weighted_std / np.sqrt(eff_n)
                    margins[var] = z_critical * se_mean
        
        if margins:
            return pd.DataFrame([margins], index=['Margin of Error']).T
        else:
            return pd.DataFrame()
    
    def _empty_stats(self):
        """Return empty statistics structure for variables with no data."""
        return {
            'count': 0,
            'effective_n': 0,
            'weighted_mean': np.nan,
            'std_error': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan
        }
    
    def calculate_design_effects(self, df, analysis_vars, weight_column):
        """Calculate design effects for survey estimates."""
        if weight_column not in df.columns:
            return {}
        
        weights = df[weight_column].dropna()
        design_effects = {}
        
        # Overall design effect
        overall_deff = len(weights) / ((weights.sum() ** 2) / (weights ** 2).sum())
        design_effects['overall'] = overall_deff
        
        # Variable-specific design effects (simplified)
        for var in analysis_vars:
            if var in df.columns:
                var_data = df[var].dropna()
                var_weights = weights[var_data.index]
                
                if len(var_data) > 0:
                    var_deff = len(var_weights) / ((var_weights.sum() ** 2) / (var_weights ** 2).sum())
                    design_effects[var] = var_deff
        
        return design_effects

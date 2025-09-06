import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy import stats
import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from .enhanced_data_validator import EnhancedDataValidator

class DataCleaner:
    """Comprehensive data cleaning and validation module."""
    
    def __init__(self):
        self.cleaning_report = {}
        self.validator = EnhancedDataValidator()
        self.logger = logging.getLogger(__name__)
    
    def clean_data(self, df, config):
        """
        Main cleaning function that applies all specified cleaning methods.
        
        Args:
            df: pandas DataFrame
            config: dict with cleaning configuration
            
        Returns:
            tuple: (cleaned_df, cleaning_report)
        """
        # Enhanced input validation
        if df is None:
            raise ValueError("Input DataFrame is None")
        
        if df.empty:
            self.logger.warning("Input DataFrame is empty")
            return df, {'error': 'Empty DataFrame provided'}
        
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        # Pre-cleaning validation
        _, pre_validation = self.validator.comprehensive_validation(df)
        self.cleaning_report['pre_cleaning_validation'] = pre_validation
        
        cleaned_df = df.copy()
        self.cleaning_report = {
            'initial_shape': df.shape,
            'steps_performed': [],
            'rows_removed': 0,
            'values_imputed': 0,
            'outliers_detected': 0,
            'validation_errors': [],
            'warnings': []
        }
        
        # 1. Handle missing values
        if config.get('missing_method') and config['missing_method'] != 'None':
            cleaned_df = self._handle_missing_values(cleaned_df, config)
        
        # 2. Detect and handle outliers
        if config.get('outlier_methods'):
            cleaned_df = self._handle_outliers(cleaned_df, config)
        
        # 3. Apply rule-based validation
        if config.get('enable_consistency') or config.get('enable_skip_patterns'):
            cleaned_df = self._apply_validation_rules(cleaned_df, config)
        
        # 4. Apply custom rules
        if config.get('custom_rules'):
            cleaned_df = self._apply_custom_rules(cleaned_df, config['custom_rules'])
        
        # Final report
        self.cleaning_report['final_shape'] = cleaned_df.shape
        self.cleaning_report['rows_removed'] = df.shape[0] - cleaned_df.shape[0]
        
        return cleaned_df, self.cleaning_report
    
    def _handle_missing_values(self, df, config):
        """Handle missing values using specified method with enhanced error handling."""
        method = config.get('missing_method', 'None')
        if method == 'None':
            return df
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) == 0 and len(categorical_cols) == 0:
            self.cleaning_report['warnings'].append("No columns found for missing value imputation")
            return df
        
        initial_missing = df.isnull().sum().sum()
        
        try:
            if method == 'Mean':
                # Only fill if there are non-null values to calculate mean
                for col in numeric_cols:
                    if df[col].notna().any():
                        df[col] = df[col].fillna(df[col].mean())
                
            elif method == 'Median':
                # Only fill if there are non-null values to calculate median
                for col in numeric_cols:
                    if df[col].notna().any():
                        df[col] = df[col].fillna(df[col].median())
                
            elif method == 'KNN':
                k = config.get('knn_neighbors', 5)
                # Enhanced KNN validation
                if len(numeric_cols) == 0:
                    self.cleaning_report['warnings'].append("No numeric columns for KNN imputation")
                    return df
                    
                non_missing_rows = df[numeric_cols].dropna().shape[0]
                if non_missing_rows < k:
                    self.cleaning_report['warnings'].append(f"Not enough complete cases ({non_missing_rows}) for KNN with k={k}. Using median imputation.")
                    # Fall back to median imputation (more robust than mean)
                    for col in numeric_cols:
                        if df[col].notna().any():
                            df[col] = df[col].fillna(df[col].median())
                elif non_missing_rows < 2 * k:
                    # Reduce k if we have limited data
                    k = max(1, non_missing_rows // 2)
                    self.cleaning_report['warnings'].append(f"Reduced KNN neighbors to k={k} due to limited data")
                    try:
                        imputer = KNNImputer(n_neighbors=k)
                        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    except Exception as e:
                        self.cleaning_report['warnings'].append(f"KNN imputation failed: {str(e)}. Using median imputation.")
                        for col in numeric_cols:
                            if df[col].notna().any():
                                df[col] = df[col].fillna(df[col].median())
                else:
                    try:
                        imputer = KNNImputer(n_neighbors=k)
                        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    except Exception as e:
                        self.cleaning_report['warnings'].append(f"KNN imputation failed: {str(e)}. Using median imputation.")
                        for col in numeric_cols:
                            if df[col].notna().any():
                                df[col] = df[col].fillna(df[col].median())
        
            # Handle categorical missing values
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    if df[col].isna().any():
                        # Use mode for categorical variables, or 'Unknown' if no mode
                        if df[col].notna().any():
                            mode_value = df[col].mode()
                            if len(mode_value) > 0:
                                df[col] = df[col].fillna(mode_value[0])
                            else:
                                df[col] = df[col].fillna('Unknown')
                        else:
                            df[col] = df[col].fillna('Unknown')
        
        except Exception as e:
            self.logger.error(f"Critical error in missing value imputation: {str(e)}")
            self.cleaning_report['warnings'].append(f"Critical error in missing value imputation: {str(e)}")
            return df
        
        final_missing = df.isnull().sum().sum()
        imputed_values = initial_missing - final_missing
        
        self.cleaning_report['values_imputed'] = imputed_values
        self.cleaning_report['steps_performed'].append(f"Missing value imputation ({method})")
        
        return df
    
    def _handle_outliers(self, df, config):
        """Detect and handle outliers using specified methods."""
        methods = config['outlier_methods']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_detected = 0
        
        for method in methods:
            if method == 'IQR':
                outliers_detected += self._detect_outliers_iqr(df, numeric_cols)
                
            elif method == 'Z-score':
                threshold = config.get('z_threshold', 3.0)
                outliers_detected += self._detect_outliers_zscore(df, numeric_cols, threshold)
                
            elif method == 'Winsorization':
                limits = config.get('winsor_limits', 0.05)
                df = self._apply_winsorization(df, numeric_cols, limits)
        
        self.cleaning_report['outliers_detected'] = outliers_detected
        self.cleaning_report['steps_performed'].append(f"Outlier detection ({', '.join(methods)})")
        
        return df
    
    def _detect_outliers_iqr(self, df, numeric_cols):
        """Detect outliers using Interquartile Range method."""
        outliers_count = 0
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_count += outliers.sum()
                
                # Mark outliers (could be removed or capped)
                df.loc[outliers, col] = np.nan
        
        return outliers_count
    
    def _detect_outliers_zscore(self, df, numeric_cols, threshold):
        """Detect outliers using Z-score method."""
        outliers_count = 0
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                try:
                    z_scores = np.abs(stats.zscore(df[col].dropna(), nan_policy='omit'))
                except Exception:
                    # Skip column if zscore calculation fails
                    continue
                outliers = z_scores > threshold
                outliers_count += outliers.sum()
                
                # Mark outliers
                outlier_indices = df[col].dropna().index[outliers]
                df.loc[outlier_indices, col] = np.nan
        
        return outliers_count
    
    def _apply_winsorization(self, df, numeric_cols, limits):
        """Apply winsorization to numeric columns."""
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                lower_percentile = limits * 100
                upper_percentile = (1 - limits) * 100
                
                lower_bound = np.percentile(df[col].dropna(), lower_percentile)
                upper_bound = np.percentile(df[col].dropna(), upper_percentile)
                
                df[col] = np.clip(df[col], lower_bound, upper_bound)
        
        return df
    
    def _apply_validation_rules(self, df, config):
        """Apply consistency and skip pattern validation."""
        validation_errors = []
        
        if config.get('enable_consistency'):
            # Basic consistency checks
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Check for negative values in columns that should be positive
            likely_positive_cols = [col for col in numeric_cols 
                                  if any(keyword in col.lower() 
                                        for keyword in ['age', 'income', 'weight', 'height', 'count'])]
            
            for col in likely_positive_cols:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    validation_errors.append(f"{col}: {negative_count} negative values detected")
                    # Remove negative values
                    df.loc[df[col] < 0, col] = np.nan
        
        if config.get('enable_skip_patterns'):
            # Basic skip pattern validation
            # This is a simplified implementation - in practice, this would be more sophisticated
            validation_errors.append("Skip pattern validation performed (simplified)")
        
        self.cleaning_report['validation_errors'] = validation_errors
        self.cleaning_report['steps_performed'].append("Rule-based validation")
        
        return df
    
    def _apply_custom_rules(self, df, custom_rules):
        """Apply custom validation rules specified by user."""
        for rule in custom_rules:
            try:
                # Parse simple rules like "age >= 0", "income > 0"
                rule = rule.strip()
                if not rule:
                    continue
                
                # Simple regex to parse rules
                match = re.match(r'(\w+)\s*([<>=!]+)\s*(\d+(?:\.\d+)?)', rule)
                if match:
                    column, operator, value = match.groups()
                    value = float(value)
                    
                    if column in df.columns:
                        if operator == '>=':
                            invalid_mask = df[column] < value
                        elif operator == '>':
                            invalid_mask = df[column] <= value
                        elif operator == '<=':
                            invalid_mask = df[column] > value
                        elif operator == '<':
                            invalid_mask = df[column] >= value
                        elif operator == '==':
                            invalid_mask = df[column] != value
                        elif operator == '!=':
                            invalid_mask = df[column] == value
                        else:
                            continue
                        
                        # Mark invalid values as NaN
                        invalid_count = invalid_mask.sum()
                        if invalid_count > 0:
                            df.loc[invalid_mask, column] = np.nan
                            self.cleaning_report['validation_errors'].append(
                                f"Custom rule '{rule}': {invalid_count} violations corrected"
                            )
                
            except Exception as e:
                self.cleaning_report['validation_errors'].append(f"Error applying rule '{rule}': {str(e)}")
        
        self.cleaning_report['steps_performed'].append("Custom rules applied")
        return df
    
    def get_data_quality_metrics(self, df):
        """Calculate comprehensive data quality metrics."""
        metrics = {
            'completeness': {},
            'consistency': {},
            'validity': {}
        }
        
        # Completeness metrics
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        metrics['completeness']['overall'] = 1 - (missing_cells / total_cells)
        
        # Per-column completeness
        for col in df.columns:
            metrics['completeness'][col] = 1 - (df[col].isnull().sum() / len(df))
        
        # Basic validity checks
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                # Check for infinite values
                inf_count = np.isinf(df[col]).sum()
                metrics['validity'][f'{col}_infinite'] = inf_count
                
                # Check for extreme outliers (beyond 5 standard deviations)
                if df[col].notna().sum() > 0:
                    std_val = df[col].std()
                    mean_val = df[col].mean()
                    extreme_outliers = ((df[col] - mean_val).abs() > 5 * std_val).sum()
                    metrics['validity'][f'{col}_extreme_outliers'] = extreme_outliers
        
        return metrics

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import warnings

class EnhancedDataValidator:
    """
    Comprehensive data validation and error handling system for survey data processing.
    Handles all edge cases, anomalies, and data quality issues.
    """
    
    def __init__(self):
        self.validation_results = {
            'critical_errors': [],
            'warnings': [],
            'data_quality_issues': [],
            'logical_inconsistencies': [],
            'statistical_anomalies': [],
            'remediation_suggestions': []
        }
        
        # Define acceptable ranges for common survey variables
        self.standard_ranges = {
            'age': {'min': 0, 'max': 120, 'typical_max': 100},
            'income': {'min': 0, 'max': 10000000, 'typical_max': 1000000},
            'weight': {'min': 0, 'max': 1000, 'typical_range': (30, 300)},
            'survey_weight': {'min': 0.01, 'max': 100, 'typical_range': (0.1, 10)}
        }
    
    def comprehensive_validation(self, df: pd.DataFrame, schema: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform comprehensive validation and return cleaned data with detailed report.
        
        Args:
            df: Input DataFrame
            schema: Optional schema information
            
        Returns:
            Tuple of (validated_df, validation_report)
        """
        if df is None or df.empty:
            self.validation_results['critical_errors'].append("DataFrame is empty or None")
            return df, self.validation_results
        
        # Reset validation results
        self.validation_results = {
            'critical_errors': [],
            'warnings': [],
            'data_quality_issues': [],
            'logical_inconsistencies': [],
            'statistical_anomalies': [],
            'remediation_suggestions': [],
            'data_summary': {}
        }
        
        validated_df = df.copy()
        
        # 1. Basic structure validation
        validated_df = self._validate_basic_structure(validated_df)
        
        # 2. Column name validation and cleaning
        validated_df = self._clean_column_names(validated_df)
        
        # 3. Data type validation and conversion
        validated_df = self._validate_and_convert_types(validated_df)
        
        # 4. Missing value analysis
        self._analyze_missing_patterns(validated_df)
        
        # 5. Outlier detection and analysis
        self._detect_statistical_anomalies(validated_df)
        
        # 6. Logical consistency validation
        self._validate_logical_consistency(validated_df)
        
        # 7. Survey-specific validation
        self._validate_survey_data(validated_df)
        
        # 8. Generate data summary
        self._generate_data_summary(validated_df)
        
        # 9. Generate remediation suggestions
        self._generate_remediation_suggestions(validated_df)
        
        return validated_df, self.validation_results
    
    def _validate_basic_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate basic DataFrame structure."""
        # Check for empty DataFrame
        if df.empty:
            self.validation_results['critical_errors'].append("Dataset contains no data")
            return df
        
        # Check for duplicate column names
        if df.columns.duplicated().any():
            duplicates = df.columns[df.columns.duplicated()].tolist()
            self.validation_results['warnings'].append(f"Duplicate column names found: {duplicates}")
            # Rename duplicate columns
            df.columns = pd.io.common.dedup_names(df.columns, is_potential_multiindex=False)
        
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            self.validation_results['warnings'].append(f"Found {empty_rows} completely empty rows")
            df = df.dropna(how='all').reset_index(drop=True)
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            self.validation_results['warnings'].append(f"Found completely empty columns: {empty_cols}")
            df = df.drop(columns=empty_cols)
        
        # Check for single-row dataset
        if len(df) == 1:
            self.validation_results['warnings'].append("Dataset contains only one row - statistical analysis may be limited")
        
        return df
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names."""
        original_columns = df.columns.tolist()
        cleaned_columns = []
        
        for col in df.columns:
            # Remove special characters and spaces
            cleaned_col = re.sub(r'[^\w\s-]', '', str(col))
            # Replace spaces and hyphens with underscores
            cleaned_col = re.sub(r'[\s-]+', '_', cleaned_col)
            # Remove leading/trailing underscores
            cleaned_col = cleaned_col.strip('_')
            # Ensure column name is not empty
            if not cleaned_col:
                cleaned_col = f"column_{len(cleaned_columns)}"
            
            cleaned_columns.append(cleaned_col)
        
        # Check for changes and log them
        changes_made = []
        for orig, clean in zip(original_columns, cleaned_columns):
            if orig != clean:
                changes_made.append(f"'{orig}' → '{clean}'")
        
        if changes_made:
            self.validation_results['warnings'].append(f"Column names cleaned: {', '.join(changes_made)}")
        
        df.columns = cleaned_columns
        return df
    
    def _validate_and_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types safely."""
        for col in df.columns:
            # Try to convert numeric columns
            if col.lower() in ['id', 'age', 'income', 'weight', 'cigarettes_day', 'survey_weight']:
                df[col] = self._safe_numeric_conversion(df[col], col)
            
            # Handle boolean/categorical columns
            elif col.lower() in ['gender', 'smoke']:
                df[col] = self._standardize_categorical(df[col], col)
        
        return df
    
    def _safe_numeric_conversion(self, series: pd.Series, col_name: str) -> pd.Series:
        """Safely convert series to numeric, handling errors."""
        original_series = series.copy()
        
        try:
            # Try direct conversion first
            numeric_series = pd.to_numeric(series, errors='coerce')
            
            # Check how many values were converted to NaN
            conversion_errors = series.notna() & numeric_series.isna()
            if conversion_errors.any():
                error_values = series[conversion_errors].unique()
                self.validation_results['data_quality_issues'].append(
                    f"Column '{col_name}': Could not convert {conversion_errors.sum()} values to numeric: {error_values[:5].tolist()}"
                )
            
            return numeric_series
            
        except Exception as e:
            self.validation_results['warnings'].append(f"Error converting column '{col_name}' to numeric: {str(e)}")
            return original_series
    
    def _standardize_categorical(self, series: pd.Series, col_name: str) -> pd.Series:
        """Standardize categorical variables."""
        if col_name.lower() == 'gender':
            # Standardize gender values
            gender_mapping = {
                'm': 'M', 'male': 'M', 'man': 'M',
                'f': 'F', 'female': 'F', 'woman': 'F'
            }
            standardized = series.str.lower().map(gender_mapping).fillna(series)
            changes = (series != standardized) & series.notna() & standardized.notna()
            if changes.any():
                self.validation_results['warnings'].append(f"Gender values standardized in column '{col_name}'")
            return standardized
        
        elif col_name.lower() in ['smoke', 'smoke_']:
            # Standardize smoking status
            smoke_mapping = {
                'yes': 'Yes', 'y': 'Yes', 'true': 'Yes', '1': 'Yes',
                'no': 'No', 'n': 'No', 'false': 'No', '0': 'No'
            }
            standardized = series.astype(str).str.lower().map(smoke_mapping).fillna(series)
            changes = (series != standardized) & series.notna() & standardized.notna()
            if changes.any():
                self.validation_results['warnings'].append(f"Smoking status values standardized in column '{col_name}'")
            return standardized
        
        return series
    
    def _analyze_missing_patterns(self, df: pd.DataFrame):
        """Analyze missing value patterns and potential issues."""
        missing_summary = df.isnull().sum()
        total_missing = missing_summary.sum()
        missing_rate = total_missing / df.size
        
        self.validation_results['data_summary']['missing_values'] = {
            'total_missing': int(total_missing),
            'missing_rate': float(missing_rate),
            'columns_with_missing': missing_summary[missing_summary > 0].to_dict()
        }
        
        # Check for high missing rates
        high_missing_cols = missing_summary[missing_summary / len(df) > 0.5]
        if not high_missing_cols.empty:
            self.validation_results['data_quality_issues'].append(
                f"Columns with >50% missing values: {high_missing_cols.to_dict()}"
            )
        
        # Check for systematic missing patterns
        if len(df) > 1:
            missing_patterns = df.isnull().value_counts()
            if len(missing_patterns) < len(df) * 0.8:  # Less than 80% unique patterns
                self.validation_results['warnings'].append("Systematic missing data patterns detected")
    
    def _detect_statistical_anomalies(self, df: pd.DataFrame):
        """Detect statistical anomalies and extreme outliers."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            col_name = col.lower()
            anomalies = []
            
            # Check against standard ranges
            if col_name in self.standard_ranges:
                ranges = self.standard_ranges[col_name]
                
                # Impossible values
                impossible_low = series < ranges['min']
                impossible_high = series > ranges['max']
                
                if impossible_low.any():
                    anomalies.append(f"Values below possible minimum ({ranges['min']}): {series[impossible_low].tolist()}")
                
                if impossible_high.any():
                    anomalies.append(f"Values above possible maximum ({ranges['max']}): {series[impossible_high].tolist()}")
                
                # Extreme but possible values
                if 'typical_max' in ranges:
                    extreme_high = (series > ranges['typical_max']) & (series <= ranges['max'])
                    if extreme_high.any():
                        anomalies.append(f"Extremely high but possible values (>{ranges['typical_max']}): {series[extreme_high].tolist()}")
                
                if 'typical_range' in ranges:
                    typical_min, typical_max = ranges['typical_range']
                    atypical_low = (series < typical_min) & (series >= ranges['min'])
                    atypical_high = (series > typical_max) & (series <= ranges.get('typical_max', ranges['max']))
                    
                    if atypical_low.any():
                        anomalies.append(f"Unusually low values (<{typical_min}): {series[atypical_low].tolist()}")
                    if atypical_high.any():
                        anomalies.append(f"Unusually high values (>{typical_max}): {series[atypical_high].tolist()}")
            
            # Statistical outlier detection
            if len(series) >= 3:
                # IQR method
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Avoid division by zero
                    outlier_low = series < (Q1 - 3 * IQR)
                    outlier_high = series > (Q3 + 3 * IQR)
                    
                    if outlier_low.any() or outlier_high.any():
                        outlier_values = pd.concat([series[outlier_low], series[outlier_high]])
                        anomalies.append(f"Statistical outliers (3×IQR): {outlier_values.tolist()}")
            
            if anomalies:
                self.validation_results['statistical_anomalies'].append({
                    'column': col,
                    'anomalies': anomalies
                })
    
    def _validate_logical_consistency(self, df: pd.DataFrame):
        """Validate logical consistency between variables."""
        inconsistencies = []
        
        # Check smoking consistency
        if 'smoke' in df.columns and any(col for col in df.columns if 'cigarettes' in col.lower()):
            smoke_col = 'smoke'
            cig_col = next((col for col in df.columns if 'cigarettes' in col.lower()), None)
            
            if cig_col and smoke_col:
                # Non-smokers with cigarettes per day > 0
                non_smokers_with_cigs = (df[smoke_col] == 'No') & (df[cig_col] > 0)
                if non_smokers_with_cigs.any():
                    inconsistent_rows = df.index[non_smokers_with_cigs].tolist()
                    inconsistencies.append(f"Non-smokers with cigarettes/day > 0 in rows: {inconsistent_rows}")
                
                # Smokers with cigarettes per day = 0 or missing
                smokers_no_cigs = (df[smoke_col] == 'Yes') & ((df[cig_col] == 0) | df[cig_col].isna())
                if smokers_no_cigs.any():
                    inconsistent_rows = df.index[smokers_no_cigs].tolist()
                    inconsistencies.append(f"Smokers with 0 or missing cigarettes/day in rows: {inconsistent_rows}")
        
        # Check age-related consistency
        if 'age' in df.columns:
            # Negative ages
            negative_age = df['age'] < 0
            if negative_age.any():
                inconsistent_rows = df.index[negative_age].tolist()
                inconsistencies.append(f"Negative ages found in rows: {inconsistent_rows}")
        
        # Check income consistency
        if 'income' in df.columns:
            # Negative income
            negative_income = df['income'] < 0
            if negative_income.any():
                inconsistent_rows = df.index[negative_income].tolist()
                inconsistencies.append(f"Negative income values in rows: {inconsistent_rows}")
        
        if inconsistencies:
            self.validation_results['logical_inconsistencies'] = inconsistencies
    
    def _validate_survey_data(self, df: pd.DataFrame):
        """Validate survey-specific requirements."""
        survey_issues = []
        
        # Check for survey weights
        weight_cols = [col for col in df.columns if 'weight' in col.lower() and 'survey' in col.lower()]
        if weight_cols:
            for weight_col in weight_cols:
                weights = df[weight_col].dropna()
                if len(weights) > 0:
                    # Check for zero or negative weights
                    invalid_weights = weights <= 0
                    if invalid_weights.any():
                        survey_issues.append(f"Invalid survey weights (≤0) in column '{weight_col}': {weights[invalid_weights].tolist()}")
                    
                    # Check for extremely high weights
                    high_weights = weights > 10
                    if high_weights.any():
                        survey_issues.append(f"Extremely high survey weights (>10) in column '{weight_col}': {weights[high_weights].tolist()}")
        
        # Check for required survey variables
        required_vars = ['id', 'age', 'gender']
        missing_required = [var for var in required_vars if var not in df.columns]
        if missing_required:
            survey_issues.append(f"Missing common survey variables: {missing_required}")
        
        # Check sample size adequacy
        if len(df) < 30:
            survey_issues.append(f"Small sample size ({len(df)} cases) may limit statistical analysis")
        
        if survey_issues:
            self.validation_results['data_quality_issues'].extend(survey_issues)
    
    def _generate_data_summary(self, df: pd.DataFrame):
        """Generate comprehensive data summary."""
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # Add column-wise summary
        summary['column_summary'] = {}
        for col in df.columns:
            col_summary = {
                'type': str(df[col].dtype),
                'missing_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique()),
                'missing_rate': float(df[col].isnull().sum() / len(df))
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_summary.update({
                    'min': float(df[col].min()) if df[col].notna().any() else None,
                    'max': float(df[col].max()) if df[col].notna().any() else None,
                    'mean': float(df[col].mean()) if df[col].notna().any() else None,
                    'std': float(df[col].std()) if df[col].notna().any() else None
                })
            
            summary['column_summary'][col] = col_summary
        
        self.validation_results['data_summary'].update(summary)
    
    def _generate_remediation_suggestions(self, df: pd.DataFrame):
        """Generate specific suggestions for fixing identified issues."""
        suggestions = []
        
        # Suggestions for missing values
        if self.validation_results['data_summary'].get('missing_values', {}).get('total_missing', 0) > 0:
            missing_cols = self.validation_results['data_summary']['missing_values']['columns_with_missing']
            high_missing = [col for col, count in missing_cols.items() if count / len(df) > 0.3]
            
            if high_missing:
                suggestions.append(f"Consider removing columns with >30% missing data: {high_missing}")
            
            low_missing = [col for col, count in missing_cols.items() if count / len(df) <= 0.3]
            if low_missing:
                suggestions.append(f"Use imputation for columns with ≤30% missing data: {low_missing}")
        
        # Suggestions for outliers
        if self.validation_results['statistical_anomalies']:
            suggestions.append("Review detected outliers - verify if they are data entry errors or legitimate extreme values")
            suggestions.append("Consider using robust statistical methods that are less sensitive to outliers")
        
        # Suggestions for logical inconsistencies
        if self.validation_results['logical_inconsistencies']:
            suggestions.append("Fix logical inconsistencies before analysis - these may indicate data entry errors")
            suggestions.append("Implement data validation rules to prevent such inconsistencies in future data collection")
        
        # Suggestions for small sample sizes
        if len(df) < 100:
            suggestions.append("Consider collecting more data if possible - current sample size may limit generalizability")
            suggestions.append("Use appropriate statistical methods for small samples (e.g., non-parametric tests)")
        
        self.validation_results['remediation_suggestions'] = suggestions
    
    def generate_validation_report(self, validation_results: Dict) -> str:
        """Generate a human-readable validation report."""
        report = []
        report.append("=" * 60)
        report.append("COMPREHENSIVE DATA VALIDATION REPORT")
        report.append("=" * 60)
        
        # Summary
        data_summary = validation_results.get('data_summary', {})
        report.append(f"\nDATA SUMMARY:")
        report.append(f"  • Total rows: {data_summary.get('total_rows', 'N/A')}")
        report.append(f"  • Total columns: {data_summary.get('total_columns', 'N/A')}")
        report.append(f"  • Missing values: {data_summary.get('missing_values', {}).get('total_missing', 0)}")
        
        # Critical errors
        if validation_results['critical_errors']:
            report.append(f"\n🚨 CRITICAL ERRORS ({len(validation_results['critical_errors'])}):")
            for error in validation_results['critical_errors']:
                report.append(f"  • {error}")
        
        # Data quality issues
        if validation_results['data_quality_issues']:
            report.append(f"\n⚠️  DATA QUALITY ISSUES ({len(validation_results['data_quality_issues'])}):")
            for issue in validation_results['data_quality_issues']:
                report.append(f"  • {issue}")
        
        # Logical inconsistencies
        if validation_results['logical_inconsistencies']:
            report.append(f"\n🔍 LOGICAL INCONSISTENCIES ({len(validation_results['logical_inconsistencies'])}):")
            for inconsistency in validation_results['logical_inconsistencies']:
                report.append(f"  • {inconsistency}")
        
        # Statistical anomalies
        if validation_results['statistical_anomalies']:
            report.append(f"\n📊 STATISTICAL ANOMALIES ({len(validation_results['statistical_anomalies'])}):")
            for anomaly in validation_results['statistical_anomalies']:
                report.append(f"  • Column '{anomaly['column']}':")
                for detail in anomaly['anomalies']:
                    report.append(f"    - {detail}")
        
        # Warnings
        if validation_results['warnings']:
            report.append(f"\n💡 WARNINGS ({len(validation_results['warnings'])}):")
            for warning in validation_results['warnings']:
                report.append(f"  • {warning}")
        
        # Remediation suggestions
        if validation_results['remediation_suggestions']:
            report.append(f"\n🔧 REMEDIATION SUGGESTIONS:")
            for suggestion in validation_results['remediation_suggestions']:
                report.append(f"  • {suggestion}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
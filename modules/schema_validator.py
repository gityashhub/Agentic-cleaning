import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional

class SchemaValidator:
    """Validate and manage data schemas for survey datasets."""
    
    def __init__(self):
        self.schema_template = {
            "version": "1.0",
            "metadata": {
                "title": "",
                "description": "",
                "created_date": "",
                "survey_type": ""
            },
            "variables": {},
            "validation_rules": [],
            "survey_design": {
                "stratification_vars": [],
                "weight_vars": [],
                "cluster_vars": []
            }
        }
    
    def auto_detect_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Automatically detect schema from DataFrame.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict: Detected schema
        """
        schema = self.schema_template.copy()
        schema["variables"] = {}
        
        for column in df.columns:
            var_schema = self._analyze_variable(df, column)
            schema["variables"][column] = var_schema
        
        # Detect potential survey design variables
        schema["survey_design"] = self._detect_survey_design_vars(df)
        
        # Generate basic validation rules
        schema["validation_rules"] = self._generate_basic_rules(df)
        
        return schema
    
    def _analyze_variable(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze a single variable and determine its properties."""
        series = df[column]
        
        var_info = {
            "name": column,
            "type": self._determine_variable_type(series),
            "description": "",
            "missing_count": series.isnull().sum(),
            "missing_rate": series.isnull().sum() / len(series),
            "unique_count": series.nunique(),
            "data_type": str(series.dtype)
        }
        
        # Add type-specific information
        if var_info["type"] == "numeric":
            var_info.update(self._analyze_numeric_variable(series))
        elif var_info["type"] == "categorical":
            var_info.update(self._analyze_categorical_variable(series))
        elif var_info["type"] == "binary":
            var_info.update(self._analyze_binary_variable(series))
        elif var_info["type"] == "date":
            var_info.update(self._analyze_date_variable(series))
        
        # Survey-specific classifications
        var_info.update(self._classify_survey_variable(column, series))
        
        return var_info
    
    def _determine_variable_type(self, series: pd.Series) -> str:
        """Determine the most appropriate variable type."""
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            # Check if binary (0/1 or True/False)
            unique_vals = series.dropna().unique()
            if len(unique_vals) == 2:
                if set(unique_vals).issubset({0, 1, True, False}):
                    return "binary"
            return "numeric"
        
        # Check if datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "date"
        
        # Check if could be converted to datetime
        if series.dtype == 'object':
            sample_values = series.dropna().head(10)
            date_like_count = 0
            for val in sample_values:
                try:
                    pd.to_datetime(str(val))
                    date_like_count += 1
                except:
                    pass
            
            if date_like_count > len(sample_values) * 0.5:
                return "date"
        
        # Check if categorical with low cardinality
        unique_count = series.nunique()
        total_count = len(series.dropna())
        
        if unique_count <= 20 or unique_count / total_count < 0.05:
            return "categorical"
        
        # Default to text
        return "text"
    
    def _analyze_numeric_variable(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric variable properties."""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {
                "min_value": None,
                "max_value": None,
                "mean": None,
                "std": None,
                "median": None,
                "quartiles": {}
            }
        
        return {
            "min_value": float(clean_series.min()),
            "max_value": float(clean_series.max()),
            "mean": float(clean_series.mean()),
            "std": float(clean_series.std()),
            "median": float(clean_series.median()),
            "quartiles": {
                "q25": float(clean_series.quantile(0.25)),
                "q75": float(clean_series.quantile(0.75))
            },
            "outlier_bounds": self._calculate_outlier_bounds(clean_series),
            "distribution_shape": self._assess_distribution_shape(clean_series)
        }
    
    def _analyze_categorical_variable(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical variable properties."""
        value_counts = series.value_counts()
        
        return {
            "categories": value_counts.index.tolist()[:20],  # Limit to top 20
            "category_counts": value_counts.head(20).to_dict(),
            "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
            "least_frequent": value_counts.index[-1] if len(value_counts) > 0 else None,
            "cardinality": len(value_counts),
            "entropy": self._calculate_entropy(value_counts)
        }
    
    def _analyze_binary_variable(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze binary variable properties."""
        value_counts = series.value_counts()
        
        return {
            "values": value_counts.index.tolist(),
            "value_counts": value_counts.to_dict(),
            "proportion_positive": value_counts.iloc[0] / value_counts.sum() if len(value_counts) > 0 else 0
        }
    
    def _analyze_date_variable(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze date variable properties."""
        try:
            date_series = pd.to_datetime(series, errors='coerce')
            clean_dates = date_series.dropna()
            
            if len(clean_dates) == 0:
                return {"date_range": None, "date_format": "unknown"}
            
            return {
                "min_date": clean_dates.min().isoformat(),
                "max_date": clean_dates.max().isoformat(),
                "date_range_days": (clean_dates.max() - clean_dates.min()).days,
                "date_format": "auto-detected"
            }
        except:
            return {"date_range": None, "date_format": "invalid"}
    
    def _classify_survey_variable(self, column_name: str, series: pd.Series) -> Dict[str, Any]:
        """Classify variable based on survey methodology conventions."""
        column_lower = column_name.lower()
        
        classification = {
            "survey_role": "response",  # response, weight, strata, cluster, id
            "measurement_level": "nominal",  # nominal, ordinal, interval, ratio
            "is_identifier": False,
            "is_weight": False,
            "is_strata": False,
            "is_demographic": False
        }
        
        # Check for common identifier patterns
        if any(keyword in column_lower for keyword in ['id', 'key', 'index', 'record']):
            classification["survey_role"] = "id"
            classification["is_identifier"] = True
        
        # Check for weight variables
        elif any(keyword in column_lower for keyword in ['weight', 'wt', 'wgt']):
            classification["survey_role"] = "weight"
            classification["is_weight"] = True
            classification["measurement_level"] = "ratio"
        
        # Check for stratification variables
        elif any(keyword in column_lower for keyword in ['strata', 'stratum', 'psu', 'cluster']):
            classification["survey_role"] = "strata"
            classification["is_strata"] = True
        
        # Check for demographic variables
        elif any(keyword in column_lower for keyword in ['age', 'sex', 'gender', 'race', 'education', 'income']):
            classification["is_demographic"] = True
            
            if 'age' in column_lower or 'income' in column_lower:
                classification["measurement_level"] = "ratio"
            elif any(keyword in column_lower for keyword in ['education', 'satisfaction']):
                classification["measurement_level"] = "ordinal"
        
        # Determine measurement level based on data characteristics
        if pd.api.types.is_numeric_dtype(series) and not classification["is_identifier"]:
            unique_count = series.nunique()
            if unique_count > 10:
                classification["measurement_level"] = "ratio"
            elif unique_count > 2:
                classification["measurement_level"] = "ordinal"
        
        return classification
    
    def _detect_survey_design_vars(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect potential survey design variables."""
        design_vars = {
            "stratification_vars": [],
            "weight_vars": [],
            "cluster_vars": []
        }
        
        for column in df.columns:
            column_lower = column.lower()
            
            # Weight variables
            if any(keyword in column_lower for keyword in ['weight', 'wt', 'wgt']):
                design_vars["weight_vars"].append(column)
            
            # Stratification variables
            elif any(keyword in column_lower for keyword in ['strata', 'stratum']):
                design_vars["stratification_vars"].append(column)
            
            # Cluster variables
            elif any(keyword in column_lower for keyword in ['psu', 'cluster', 'primary']):
                design_vars["cluster_vars"].append(column)
            
            # Geographic variables that might be used for stratification
            elif any(keyword in column_lower for keyword in ['region', 'state', 'province', 'district']):
                if df[column].nunique() < 50:  # Reasonable number of strata
                    design_vars["stratification_vars"].append(column)
        
        return design_vars
    
    def _generate_basic_rules(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate basic validation rules."""
        rules = []
        
        for column in df.columns:
            series = df[column]
            column_lower = column.lower()
            
            # Age rules
            if 'age' in column_lower and pd.api.types.is_numeric_dtype(series):
                rules.append({
                    "variable": column,
                    "rule_type": "range",
                    "min_value": 0,
                    "max_value": 120,
                    "description": "Age should be between 0 and 120"
                })
            
            # Income rules
            elif 'income' in column_lower and pd.api.types.is_numeric_dtype(series):
                rules.append({
                    "variable": column,
                    "rule_type": "range",
                    "min_value": 0,
                    "max_value": None,
                    "description": "Income should be non-negative"
                })
            
            # Weight rules
            elif any(keyword in column_lower for keyword in ['weight', 'wt', 'wgt']):
                rules.append({
                    "variable": column,
                    "rule_type": "range",
                    "min_value": 0,
                    "max_value": None,
                    "description": "Survey weights should be positive"
                })
            
            # Completeness rules for key variables
            if any(keyword in column_lower for keyword in ['id', 'key', 'record']):
                rules.append({
                    "variable": column,
                    "rule_type": "completeness",
                    "min_completeness": 1.0,
                    "description": "Identifier variables should be complete"
                })
        
        return rules
    
    def validate_data_against_schema(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate DataFrame against schema."""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "variable_results": {}
        }
        
        # Check if all schema variables exist in data
        schema_vars = set(schema.get("variables", {}).keys())
        data_vars = set(df.columns)
        
        missing_vars = schema_vars - data_vars
        extra_vars = data_vars - schema_vars
        
        if missing_vars:
            validation_results["errors"].append(f"Missing variables: {list(missing_vars)}")
            validation_results["is_valid"] = False
        
        if extra_vars:
            validation_results["warnings"].append(f"Extra variables not in schema: {list(extra_vars)}")
        
        # Validate each variable
        for var_name, var_schema in schema.get("variables", {}).items():
            if var_name in df.columns:
                var_result = self._validate_variable(df[var_name], var_schema)
                validation_results["variable_results"][var_name] = var_result
                
                if not var_result["is_valid"]:
                    validation_results["is_valid"] = False
        
        # Apply validation rules
        for rule in schema.get("validation_rules", []):
            rule_result = self._apply_validation_rule(df, rule)
            if not rule_result["is_valid"]:
                validation_results["errors"].extend(rule_result["errors"])
                validation_results["is_valid"] = False
        
        return validation_results
    
    def _validate_variable(self, series: pd.Series, var_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single variable against its schema."""
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        expected_type = var_schema.get("type")
        actual_type = self._determine_variable_type(series)
        
        if expected_type and expected_type != actual_type:
            result["errors"].append(f"Type mismatch: expected {expected_type}, got {actual_type}")
            result["is_valid"] = False
        
        # Check numeric ranges
        if expected_type == "numeric" and "min_value" in var_schema:
            min_val = var_schema["min_value"]
            if min_val is not None:
                violations = (series < min_val).sum()
                if violations > 0:
                    result["errors"].append(f"{violations} values below minimum {min_val}")
                    result["is_valid"] = False
        
        if expected_type == "numeric" and "max_value" in var_schema:
            max_val = var_schema["max_value"]
            if max_val is not None:
                violations = (series > max_val).sum()
                if violations > 0:
                    result["errors"].append(f"{violations} values above maximum {max_val}")
                    result["is_valid"] = False
        
        # Check categorical values
        if expected_type == "categorical" and "categories" in var_schema:
            expected_cats = set(var_schema["categories"])
            actual_cats = set(series.dropna().unique())
            invalid_cats = actual_cats - expected_cats
            
            if invalid_cats:
                result["errors"].append(f"Invalid categories: {list(invalid_cats)}")
                result["is_valid"] = False
        
        return result
    
    def _apply_validation_rule(self, df: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a single validation rule."""
        result = {
            "is_valid": True,
            "errors": []
        }
        
        variable = rule.get("variable")
        rule_type = rule.get("rule_type")
        
        if variable not in df.columns:
            result["errors"].append(f"Variable {variable} not found")
            result["is_valid"] = False
            return result
        
        series = df[variable]
        
        if rule_type == "range":
            min_val = rule.get("min_value")
            max_val = rule.get("max_value")
            
            if min_val is not None:
                violations = (series < min_val).sum()
                if violations > 0:
                    result["errors"].append(f"{variable}: {violations} values below {min_val}")
                    result["is_valid"] = False
            
            if max_val is not None:
                violations = (series > max_val).sum()
                if violations > 0:
                    result["errors"].append(f"{variable}: {violations} values above {max_val}")
                    result["is_valid"] = False
        
        elif rule_type == "completeness":
            min_completeness = rule.get("min_completeness", 1.0)
            actual_completeness = 1 - (series.isnull().sum() / len(series))
            
            if actual_completeness < min_completeness:
                result["errors"].append(
                    f"{variable}: completeness {actual_completeness:.3f} below required {min_completeness:.3f}"
                )
                result["is_valid"] = False
        
        return result
    
    def _calculate_outlier_bounds(self, series: pd.Series) -> Dict[str, float]:
        """Calculate outlier bounds using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        return {
            "lower_bound": float(Q1 - 1.5 * IQR),
            "upper_bound": float(Q3 + 1.5 * IQR)
        }
    
    def _assess_distribution_shape(self, series: pd.Series) -> str:
        """Assess the shape of the distribution."""
        try:
            from scipy import stats
            skewness = stats.skew(series)
            
            if abs(skewness) < 0.5:
                return "normal"
            elif skewness > 0.5:
                return "right_skewed"
            else:
                return "left_skewed"
        except:
            return "unknown"
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy of categorical distribution."""
        proportions = value_counts / value_counts.sum()
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
        return float(entropy)
    
    def export_schema(self, schema: Dict[str, Any], filename: str) -> str:
        """Export schema to JSON file."""
        with open(filename, 'w') as f:
            json.dump(schema, f, indent=2, default=str)
        return filename
    
    def import_schema(self, filename: str) -> Dict[str, Any]:
        """Import schema from JSON file."""
        with open(filename, 'r') as f:
            schema = json.load(f)
        return schema

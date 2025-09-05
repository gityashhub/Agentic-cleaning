import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional, Union
import io
import base64
from datetime import datetime
import re

class DataValidationHelpers:
    """Helper functions for data validation and quality checks."""
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict: Data quality metrics and issues
        """
        quality_report = {
            'overall_score': 0,
            'issues': [],
            'warnings': [],
            'metrics': {},
            'recommendations': []
        }
        
        # Basic metrics
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        quality_report['metrics'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_rate': missing_cells / total_cells if total_cells > 0 else 0,
            'duplicate_rate': duplicate_rows / len(df) if len(df) > 0 else 0,
            'completeness': 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        }
        
        # Quality scoring (0-100)
        completeness_score = quality_report['metrics']['completeness'] * 40
        duplicate_score = (1 - quality_report['metrics']['duplicate_rate']) * 30
        consistency_score = DataValidationHelpers._assess_consistency(df) * 30
        
        quality_report['overall_score'] = completeness_score + duplicate_score + consistency_score
        
        # Identify issues
        if quality_report['metrics']['missing_rate'] > 0.3:
            quality_report['issues'].append("High missing data rate (>30%)")
        elif quality_report['metrics']['missing_rate'] > 0.1:
            quality_report['warnings'].append("Moderate missing data rate (>10%)")
        
        if quality_report['metrics']['duplicate_rate'] > 0.05:
            quality_report['issues'].append("High duplicate rate (>5%)")
        
        # Generate recommendations
        if quality_report['metrics']['missing_rate'] > 0.1:
            quality_report['recommendations'].append("Consider imputation strategies for missing data")
        
        if duplicate_rows > 0:
            quality_report['recommendations'].append("Review and remove duplicate records")
        
        return quality_report
    
    @staticmethod
    def _assess_consistency(df: pd.DataFrame) -> float:
        """Assess data consistency across columns."""
        consistency_score = 1.0
        
        # Check for inconsistent data types within text columns
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            # Check for mixed case inconsistencies
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                # Simple consistency check - could be expanded
                unique_lower = non_null_values.str.lower().nunique()
                unique_original = non_null_values.nunique()
                
                if unique_original > unique_lower:
                    consistency_score -= 0.1
        
        return max(0, consistency_score)
    
    @staticmethod
    def identify_potential_identifiers(df: pd.DataFrame) -> List[str]:
        """Identify columns that might be unique identifiers."""
        potential_ids = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check naming patterns
            if any(keyword in col_lower for keyword in ['id', 'key', 'index', 'record', 'seq']):
                potential_ids.append(col)
                continue
            
            # Check uniqueness
            if df[col].nunique() == len(df) and df[col].nunique() > 1:
                potential_ids.append(col)
        
        return potential_ids
    
    @staticmethod
    def suggest_data_types(df: pd.DataFrame) -> Dict[str, str]:
        """Suggest optimal data types for columns."""
        suggestions = {}
        
        for col in df.columns:
            current_type = str(df[col].dtype)
            
            # Skip if already optimal
            if current_type in ['int64', 'float64', 'datetime64[ns]', 'bool']:
                continue
            
            # Try to infer better types
            non_null_data = df[col].dropna()
            
            if len(non_null_data) == 0:
                continue
            
            # Check if can be converted to numeric
            try:
                pd.to_numeric(non_null_data)
                suggestions[col] = 'numeric'
                continue
            except:
                pass
            
            # Check if can be converted to datetime
            try:
                pd.to_datetime(non_null_data)
                suggestions[col] = 'datetime'
                continue
            except:
                pass
            
            # Check if should be categorical
            if non_null_data.nunique() / len(non_null_data) < 0.05:
                suggestions[col] = 'category'
        
        return suggestions

class StatisticalHelpers:
    """Helper functions for statistical computations."""
    
    @staticmethod
    def calculate_effective_sample_size(weights: pd.Series) -> float:
        """Calculate effective sample size from survey weights."""
        if len(weights) == 0:
            return 0
        
        return (weights.sum() ** 2) / (weights ** 2).sum()
    
    @staticmethod
    def calculate_design_effect(weights: pd.Series) -> float:
        """Calculate design effect from survey weights."""
        eff_n = StatisticalHelpers.calculate_effective_sample_size(weights)
        if eff_n == 0:
            return 1.0
        
        return len(weights) / eff_n
    
    @staticmethod
    def calculate_confidence_interval(
        estimate: float, 
        std_error: float, 
        confidence_level: float = 0.95,
        df: Optional[int] = None
    ) -> tuple:
        """Calculate confidence interval for an estimate."""
        from scipy import stats
        
        alpha = 1 - confidence_level
        
        if df is not None and df > 0:
            # Use t-distribution
            t_critical = stats.t.ppf(1 - alpha/2, df)
            margin = t_critical * std_error
        else:
            # Use normal distribution
            z_critical = stats.norm.ppf(1 - alpha/2)
            margin = z_critical * std_error
        
        return (estimate - margin, estimate + margin)
    
    @staticmethod
    def calculate_weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
        """Calculate weighted quantile."""
        if len(values) == 0 or len(weights) == 0:
            return np.nan
        
        # Remove NaN values
        mask = ~(values.isna() | weights.isna())
        clean_values = values[mask]
        clean_weights = weights[mask]
        
        if len(clean_values) == 0:
            return np.nan
        
        # Sort by values
        sorted_indices = np.argsort(clean_values)
        sorted_values = clean_values.iloc[sorted_indices]
        sorted_weights = clean_weights.iloc[sorted_indices]
        
        # Calculate cumulative weights
        cumulative_weights = np.cumsum(sorted_weights)
        total_weight = cumulative_weights[-1]
        
        # Find the quantile position
        target_weight = quantile * total_weight
        
        # Find the value at the target weight
        idx = np.searchsorted(cumulative_weights, target_weight)
        
        if idx >= len(sorted_values):
            return sorted_values.iloc[-1]
        elif idx == 0:
            return sorted_values.iloc[0]
        else:
            return sorted_values.iloc[idx]
    
    @staticmethod
    def perform_outlier_detection(
        series: pd.Series, 
        method: str = 'iqr',
        **kwargs
    ) -> pd.Series:
        """Detect outliers using specified method."""
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            multiplier = kwargs.get('multiplier', 1.5)
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == 'zscore':
            from scipy import stats
            threshold = kwargs.get('threshold', 3)
            z_scores = np.abs(stats.zscore(series.dropna()))
            
            # Create boolean series with same index as original
            outliers = pd.Series(False, index=series.index)
            outliers.loc[series.dropna().index] = z_scores > threshold
            
            return outliers
        
        elif method == 'modified_zscore':
            median = series.median()
            mad = np.median(np.abs(series - median))
            threshold = kwargs.get('threshold', 3.5)
            
            modified_z = 0.6745 * (series - median) / mad
            return np.abs(modified_z) > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

class UIHelpers:
    """Helper functions for Streamlit UI components."""
    
    @staticmethod
    def create_download_link(data: bytes, filename: str, mime_type: str, link_text: str) -> str:
        """Create a download link for data."""
        b64_data = base64.b64encode(data).decode()
        href = f'<a href="data:{mime_type};base64,{b64_data}" download="{filename}">{link_text}</a>'
        return href
    
    @staticmethod
    def display_metric_card(title: str, value: Any, delta: Optional[str] = None, help_text: Optional[str] = None):
        """Display a metric in a card format."""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.metric(
                label=title,
                value=value,
                delta=delta,
                help=help_text
            )
    
    @staticmethod
    def create_progress_indicator(current_step: int, total_steps: int, step_names: List[str]):
        """Create a progress indicator for multi-step processes."""
        st.markdown("### Processing Progress")
        
        progress = current_step / total_steps
        st.progress(progress)
        
        # Show current step
        if current_step <= len(step_names):
            current_step_name = step_names[current_step - 1]
            st.info(f"Current Step: {current_step_name} ({current_step}/{total_steps})")
        
        # Show completed steps
        with st.expander("View All Steps"):
            for i, step_name in enumerate(step_names, 1):
                if i < current_step:
                    st.success(f"✅ {step_name}")
                elif i == current_step:
                    st.info(f"🔄 {step_name} (In Progress)")
                else:
                    st.write(f"⏳ {step_name}")
    
    @staticmethod
    def display_data_summary_cards(df: pd.DataFrame):
        """Display data summary in card format."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{len(df):,}")
        
        with col2:
            st.metric("Columns", len(df.columns))
        
        with col3:
            missing_pct = (df.isnull().sum().sum() / df.size) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        with col4:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    @staticmethod
    def create_variable_selector(
        df: pd.DataFrame, 
        label: str,
        variable_types: Optional[List[str]] = None,
        key: Optional[str] = None
    ) -> List[str]:
        """Create a smart variable selector with filtering options."""
        all_columns = df.columns.tolist()
        
        if variable_types:
            filtered_columns = []
            
            for col in all_columns:
                col_type = UIHelpers._get_variable_type(df[col])
                if col_type in variable_types:
                    filtered_columns.append(col)
        else:
            filtered_columns = all_columns
        
        return st.multiselect(
            label,
            options=filtered_columns,
            key=key,
            help=f"Select from {len(filtered_columns)} available variables"
        )
    
    @staticmethod
    def _get_variable_type(series: pd.Series) -> str:
        """Get simplified variable type for filtering."""
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        elif series.nunique() < 20:
            return 'categorical'
        else:
            return 'text'

class FileHelpers:
    """Helper functions for file operations."""
    
    @staticmethod
    def validate_file_upload(uploaded_file) -> Dict[str, Any]:
        """Validate uploaded file and return metadata."""
        if uploaded_file is None:
            return {'valid': False, 'error': 'No file uploaded'}
        
        file_info = {
            'valid': True,
            'filename': uploaded_file.name,
            'size': uploaded_file.size,
            'type': uploaded_file.type,
            'extension': uploaded_file.name.split('.')[-1].lower()
        }
        
        # Size validation (max 500MB)
        max_size = 500 * 1024 * 1024
        if file_info['size'] > max_size:
            file_info['valid'] = False
            file_info['error'] = f"File size ({file_info['size']/1024**2:.1f} MB) exceeds limit (500 MB)"
        
        # Extension validation
        allowed_extensions = ['csv', 'xlsx', 'xls']
        if file_info['extension'] not in allowed_extensions:
            file_info['valid'] = False
            file_info['error'] = f"File type '{file_info['extension']}' not supported. Allowed: {allowed_extensions}"
        
        return file_info
    
    @staticmethod
    def create_sample_data_template() -> pd.DataFrame:
        """Create a sample data template for users."""
        sample_data = {
            'respondent_id': range(1, 101),
            'age': np.random.randint(18, 80, 100),
            'gender': np.random.choice(['Male', 'Female', 'Other'], 100),
            'income': np.random.normal(50000, 15000, 100).round(0),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
            'satisfaction': np.random.randint(1, 6, 100),
            'weight': np.random.uniform(0.5, 2.0, 100).round(3)
        }
        
        return pd.DataFrame(sample_data)
    
    @staticmethod
    def export_to_excel(data_dict: Dict[str, pd.DataFrame], filename: str) -> bytes:
        """Export multiple DataFrames to Excel with separate sheets."""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                # Clean sheet name (Excel limitations)
                clean_name = re.sub(r'[\\/*?:"<>|]', '', sheet_name)[:31]
                df.to_excel(writer, sheet_name=clean_name, index=False)
        
        output.seek(0)
        return output.getvalue()

class TextHelpers:
    """Helper functions for text processing and formatting."""
    
    @staticmethod
    def format_number(value: Union[int, float], decimal_places: int = 2) -> str:
        """Format numbers for display with proper thousand separators."""
        if pd.isna(value):
            return "N/A"
        
        if isinstance(value, int) or value.is_integer():
            return f"{int(value):,}"
        else:
            return f"{value:,.{decimal_places}f}"
    
    @staticmethod
    def format_percentage(value: float, decimal_places: int = 1) -> str:
        """Format percentage values."""
        if pd.isna(value):
            return "N/A"
        
        return f"{value * 100:.{decimal_places}f}%"
    
    @staticmethod
    def generate_variable_description(column_name: str, series: pd.Series) -> str:
        """Generate automatic description for variables."""
        desc_parts = []
        
        # Basic info
        var_type = UIHelpers._get_variable_type(series)
        desc_parts.append(f"{var_type.title()} variable")
        
        # Missing data info
        missing_pct = (series.isnull().sum() / len(series)) * 100
        if missing_pct > 0:
            desc_parts.append(f"{missing_pct:.1f}% missing")
        
        # Type-specific info
        if var_type == 'numeric':
            desc_parts.append(f"Range: {series.min():.2f} to {series.max():.2f}")
        elif var_type == 'categorical':
            desc_parts.append(f"{series.nunique()} categories")
        
        return f"{column_name}: " + ", ".join(desc_parts)
    
    @staticmethod
    def clean_column_name(name: str) -> str:
        """Clean column names for better display."""
        # Replace underscores and camelCase with spaces
        cleaned = re.sub(r'[_]', ' ', name)
        cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)
        
        # Title case
        return cleaned.title()

class ConfigHelpers:
    """Helper functions for configuration management."""
    
    @staticmethod
    def save_processing_config(config: Dict[str, Any]) -> str:
        """Save processing configuration to session state."""
        config_key = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if 'processing_configs' not in st.session_state:
            st.session_state.processing_configs = {}
        
        st.session_state.processing_configs[config_key] = config
        return config_key
    
    @staticmethod
    def load_processing_config(config_key: str) -> Optional[Dict[str, Any]]:
        """Load processing configuration from session state."""
        if 'processing_configs' not in st.session_state:
            return None
        
        return st.session_state.processing_configs.get(config_key)
    
    @staticmethod
    def get_default_cleaning_config() -> Dict[str, Any]:
        """Get default data cleaning configuration."""
        return {
            'missing_method': 'Mean',
            'outlier_methods': ['IQR'],
            'enable_consistency': True,
            'enable_skip_patterns': True,
            'knn_neighbors': 5,
            'z_threshold': 3.0,
            'winsor_limits': 0.05,
            'trim_weights': False,
            'normalize_weights': True
        }
    
    @staticmethod
    def validate_config(config: Dict[str, Any], config_type: str) -> Dict[str, Any]:
        """Validate configuration parameters."""
        validation_result = {'valid': True, 'errors': [], 'warnings': []}
        
        if config_type == 'cleaning':
            # Validate cleaning configuration
            if config.get('knn_neighbors', 5) < 1:
                validation_result['errors'].append("KNN neighbors must be >= 1")
                validation_result['valid'] = False
            
            if config.get('z_threshold', 3.0) < 1:
                validation_result['errors'].append("Z-score threshold must be >= 1")
                validation_result['valid'] = False
            
            winsor_limits = config.get('winsor_limits', 0.05)
            if not (0 < winsor_limits < 0.5):
                validation_result['errors'].append("Winsorization limits must be between 0 and 0.5")
                validation_result['valid'] = False
        
        return validation_result

class ErrorHandling:
    """Helper functions for error handling and user messaging."""
    
    @staticmethod
    def handle_processing_error(error: Exception, context: str) -> None:
        """Handle processing errors with user-friendly messages."""
        error_type = type(error).__name__
        error_message = str(error)
        
        st.error(f"❌ Error in {context}")
        
        # Provide specific guidance based on error type
        if error_type == "MemoryError":
            st.error("**Memory Error:** The dataset is too large for processing. Try reducing the dataset size or using a machine with more memory.")
        
        elif error_type == "KeyError":
            st.error(f"**Column Not Found:** {error_message}. Please check that all required columns exist in your data.")
        
        elif error_type == "ValueError":
            st.error(f"**Invalid Data:** {error_message}. Please check your data format and try again.")
        
        elif "permission" in error_message.lower():
            st.error("**Permission Error:** Unable to access the file. Please check file permissions and try again.")
        
        else:
            st.error(f"**{error_type}:** {error_message}")
        
        # Show expandable details for debugging
        with st.expander("Technical Details"):
            st.code(f"""
Error Type: {error_type}
Context: {context}
Message: {error_message}
            """)
    
    @staticmethod
    def display_warning(message: str, category: str = "general") -> None:
        """Display categorized warnings to users."""
        warning_icons = {
            "data_quality": "⚠️",
            "performance": "🐌",
            "methodology": "📊",
            "general": "ℹ️"
        }
        
        icon = warning_icons.get(category, "ℹ️")
        st.warning(f"{icon} {message}")
    
    @staticmethod
    def validate_user_input(input_value: Any, input_type: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Validate user input with specific constraints."""
        result = {'valid': True, 'message': ''}
        
        if input_type == 'numeric':
            try:
                value = float(input_value)
                
                if 'min_value' in constraints and value < constraints['min_value']:
                    result['valid'] = False
                    result['message'] = f"Value must be >= {constraints['min_value']}"
                
                if 'max_value' in constraints and value > constraints['max_value']:
                    result['valid'] = False
                    result['message'] = f"Value must be <= {constraints['max_value']}"
                    
            except (ValueError, TypeError):
                result['valid'] = False
                result['message'] = "Must be a valid number"
        
        elif input_type == 'list':
            if not isinstance(input_value, list) or len(input_value) == 0:
                result['valid'] = False
                result['message'] = "Must select at least one option"
        
        return result

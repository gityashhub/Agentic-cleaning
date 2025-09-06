import pandas as pd
import streamlit as st
from io import StringIO
import chardet
import logging
from typing import Tuple, Dict, Any
from .enhanced_data_validator import EnhancedDataValidator

class DataLoader:
    """Handles loading of CSV and Excel files with various encodings and formats."""
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls']
        self.validator = EnhancedDataValidator()
        self.logger = logging.getLogger(__name__)
    
    def load_file(self, uploaded_file) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load a file and return a pandas DataFrame with validation results.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (loaded_data, validation_results)
        """
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        try:
            if file_extension == 'csv':
                df = self._load_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = self._load_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Comprehensive validation
            validated_df, validation_results = self.validator.comprehensive_validation(df)
            
            return validated_df, validation_results
            
        except Exception as e:
            self.logger.error(f"Error loading {file_extension.upper()} file: {str(e)}")
            raise Exception(f"Error loading {file_extension.upper()} file: {str(e)}")
    
    def _load_csv(self, uploaded_file):
        """Load CSV file with encoding detection."""
        # Read the file content as bytes
        content = uploaded_file.read()
        
        # Detect encoding
        encoding_result = chardet.detect(content)
        encoding = encoding_result['encoding'] or 'utf-8'
        
        # Convert to string
        string_data = content.decode(encoding)
        
        # Try different delimiters
        delimiters = [',', ';', '\t', '|']
        
        for delimiter in delimiters:
            try:
                df = pd.read_csv(StringIO(string_data), delimiter=delimiter)
                
                # Check if the DataFrame makes sense (more than 1 column, reasonable number of rows)
                if df.shape[1] > 1 and df.shape[0] > 0:
                    return df
            except:
                continue
        
        # Fallback to default comma delimiter
        return pd.read_csv(StringIO(string_data))
    
    def _load_excel(self, uploaded_file):
        """Load Excel file."""
        # Read Excel file
        excel_file = pd.ExcelFile(uploaded_file)
        
        # If multiple sheets, use the first one by default
        sheet_name = excel_file.sheet_names[0]
        
        # Load the data
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        
        return df
    
    def get_file_info(self, uploaded_file):
        """Get basic information about the uploaded file."""
        return {
            'filename': uploaded_file.name,
            'size': uploaded_file.size,
            'type': uploaded_file.type
        }
    
    def validate_data_structure(self, df):
        """
        Validate the basic structure of loaded data.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict: Validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for empty DataFrame
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Dataset is empty")
            return validation_results
        
        # Check for duplicate column names
        if df.columns.duplicated().any():
            validation_results['warnings'].append("Duplicate column names detected")
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            validation_results['warnings'].append(f"Empty columns detected: {empty_cols}")
        
        # Check for extremely high missing data rate
        missing_rate = df.isnull().sum().sum() / df.size
        if missing_rate > 0.8:
            validation_results['warnings'].append(f"High missing data rate: {missing_rate:.1%}")
        
        # Check for very wide datasets (might indicate transposed data)
        if df.shape[1] > df.shape[0] and df.shape[1] > 100:
            validation_results['warnings'].append("Dataset is very wide - check if data is transposed")
        
        return validation_results

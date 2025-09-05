# Survey Data Processing Platform

## Project Overview
This is a comprehensive Streamlit-based web application for survey data processing designed for statistical agencies. The platform provides AI-augmented data cleaning, weighting, and automated report generation capabilities.

## Current State
- ✅ **Environment Setup**: Python 3.11 with all dependencies installed
- ✅ **Application Running**: Streamlit app running on port 5000
- ✅ **Configuration**: Configured for Replit environment with proxy support
- ✅ **Deployment**: Set up for autoscale deployment target

## Architecture
The application follows a modular structure:

### Main Components
- **app.py**: Main Streamlit application with multi-page interface
- **modules/**: Core processing modules
  - `data_loader.py`: Handles CSV/Excel file loading with encoding detection
  - `data_cleaner.py`: Comprehensive data cleaning and validation
  - `weight_calculator.py`: Survey weight application and statistical computation
  - `report_generator.py`: Automated report generation (PDF/HTML)
  - `visualizations.py`: Data quality and analysis visualizations
  - `schema_validator.py`: Data schema validation and mapping
- **templates/**: Report templates
- **utils/**: Helper utilities

### Features
1. **Data Upload & Schema Configuration**
   - CSV/Excel file upload with automatic encoding detection
   - Schema auto-detection and manual configuration
   - Data preview and basic statistics

2. **Data Cleaning & Validation**
   - Missing value imputation (Mean, Median, KNN)
   - Outlier detection (IQR, Z-score, Winsorization)
   - Rule-based validation and consistency checks
   - Custom validation rules

3. **Weight Application & Statistical Computation**
   - Survey weight application with diagnostics
   - Weight trimming and normalization
   - Stratified analysis support
   - Confidence interval calculations

4. **Analysis & Visualization**
   - Interactive data quality visualizations
   - Missing data pattern analysis
   - Distribution analysis
   - Statistical results visualization

5. **Report Generation**
   - Automated PDF and HTML report generation
   - Multiple template styles (Standard, Executive, Technical)
   - Custom section support
   - Methodology and diagnostics inclusion

## Configuration
- **Streamlit Config**: Located in `.streamlit/config.toml`
  - Configured for Replit proxy environment
  - CORS enabled for iframe compatibility
  - File upload limit set to 200MB
  - Custom theme applied

## Deployment
- **Development**: Streamlit dev server on port 5000
- **Production**: Configured for autoscale deployment
- **Environment**: Optimized for cloud deployment with Replit

## Technical Requirements
- Python 3.11+
- Streamlit ≥1.49.1
- Scientific computing stack (pandas, numpy, scipy, scikit-learn)
- Visualization libraries (plotly, seaborn, matplotlib)
- Report generation (reportlab)
- Data handling (openpyxl, chardet)

## Recent Changes
- **2025-09-05**: Project imported from GitHub and configured for Replit environment
  - Installed Python 3.11 and all required dependencies (streamlit, pandas, numpy, scipy, etc.)
  - Fixed Streamlit configuration for Replit proxy environment with CORS and XSRF disabled
  - Updated .streamlit/config.toml with proper server settings (port 5000, address 0.0.0.0)
  - Set up development workflow running Streamlit on port 5000 with proxy support
  - Configured autoscale deployment target for production
  - Verified application functionality and server responsiveness

## User Preferences
- No specific preferences recorded yet

## Next Steps
- The application is ready for use and can process survey data through its complete workflow
- Users can upload data files and proceed through the guided process
- All core functionality is operational and tested
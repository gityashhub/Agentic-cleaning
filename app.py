import streamlit as st
import pandas as pd
import numpy as np
from modules.data_loader import DataLoader
from modules.data_cleaner import DataCleaner
from modules.weight_calculator import WeightCalculator
from modules.report_generator import ReportGenerator
from modules.visualizations import Visualizations
from modules.schema_validator import SchemaValidator
import json

# Page configuration
st.set_page_config(
    page_title="Survey Data Processing Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'schema' not in st.session_state:
    st.session_state.schema = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'weighted_results' not in st.session_state:
    st.session_state.weighted_results = None
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []

def main():
    st.title("📊 Survey Data Processing Platform")
    st.markdown("*AI-augmented data cleaning, weighting, and automated report generation for statistical agencies*")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Processing Step",
        [
            "📁 Data Upload & Schema",
            "🧹 Data Cleaning",
            "⚖️ Weight Application", 
            "📈 Analysis & Visualization",
            "📄 Report Generation",
            "📋 Processing Log"
        ]
    )
    
    # Help section
    with st.sidebar.expander("ℹ️ Help & Guidance"):
        st.markdown("""
        **Processing Workflow:**
        1. Upload CSV/Excel data
        2. Map data schema
        3. Clean and validate data
        4. Apply survey weights
        5. Generate analysis reports
        
        **Need Help?**
        - Hover over ℹ️ icons for tooltips
        - Check processing log for details
        - Error messages provide guidance
        """)
    
    # Route to selected page
    if page == "📁 Data Upload & Schema":
        data_upload_page()
    elif page == "🧹 Data Cleaning":
        data_cleaning_page()
    elif page == "⚖️ Weight Application":
        weight_application_page()
    elif page == "📈 Analysis & Visualization":
        analysis_page()
    elif page == "📄 Report Generation":
        report_generation_page()
    elif page == "📋 Processing Log":
        processing_log_page()

def data_upload_page():
    st.header("📁 Data Upload & Schema Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("File Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your raw survey data file. Supported formats: CSV, Excel"
        )
        
        if uploaded_file is not None:
            try:
                loader = DataLoader()
                data = loader.load_file(uploaded_file)
                st.session_state.data = data
                
                st.success(f"✅ File loaded successfully! Shape: {data.shape}")
                
                # Data preview
                st.subheader("Data Preview")
                st.dataframe(data.head(10), width="stretch")
                
                # Basic statistics
                st.subheader("Basic Statistics")
                col_stats1, col_stats2 = st.columns(2)
                
                with col_stats1:
                    st.metric("Rows", f"{data.shape[0]:,}")
                    st.metric("Numeric Columns", len(data.select_dtypes(include=[np.number]).columns))
                
                with col_stats2:
                    st.metric("Columns", data.shape[1])
                    st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                return
    
    with col2:
        st.subheader("Schema Configuration")
        
        if st.session_state.data is not None:
            # Schema mapping options
            schema_method = st.radio(
                "Schema Method",
                ["Auto-detect", "Manual Configuration", "JSON Upload"],
                help="Choose how to configure your data schema"
            )
            
            if schema_method == "Auto-detect":
                if st.button("🔍 Auto-detect Schema"):
                    validator = SchemaValidator()
                    schema = validator.auto_detect_schema(st.session_state.data)
                    st.session_state.schema = schema
                    st.success("✅ Schema auto-detected!")
                    
            elif schema_method == "Manual Configuration":
                st.info("Configure column types and survey parameters manually")
                # Manual schema configuration interface would go here
                
            elif schema_method == "JSON Upload":
                schema_file = st.file_uploader(
                    "Upload Schema JSON",
                    type=['json'],
                    help="Upload a pre-configured schema file"
                )
                if schema_file is not None:
                    try:
                        schema = json.load(schema_file)
                        st.session_state.schema = schema
                        st.success("✅ Schema loaded from JSON!")
                    except Exception as e:
                        st.error(f"❌ Invalid JSON schema: {str(e)}")
        
        # Display current schema
        if st.session_state.schema:
            st.subheader("Current Schema")
            st.json(st.session_state.schema)

def data_cleaning_page():
    st.header("🧹 Data Cleaning & Validation")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first in the Data Upload page.")
        return
    
    cleaner = DataCleaner()
    
    # Cleaning configuration
    st.subheader("Cleaning Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Missing Value Imputation**")
        missing_method = st.selectbox(
            "Imputation Method",
            ["None", "Mean", "Median", "KNN"],
            help="Choose method for handling missing values"
        )
        
        knn_neighbors = 5  # Default value
        if missing_method == "KNN":
            knn_neighbors = st.slider("Number of neighbors", 3, 15, 5)
        
        st.markdown("**Outlier Detection**")
        outlier_methods = st.multiselect(
            "Detection Methods",
            ["IQR", "Z-score", "Winsorization"],
            default=["IQR"],
            help="Select outlier detection methods to apply"
        )
        
        z_threshold = 3.0  # Default value
        if "Z-score" in outlier_methods:
            z_threshold = st.slider("Z-score threshold", 2.0, 4.0, 3.0)
        
        winsor_limits = 0.05  # Default value
        if "Winsorization" in outlier_methods:
            winsor_limits = st.slider("Winsorization limits", 0.01, 0.1, 0.05)
    
    with col2:
        st.markdown("**Rule-based Validation**")
        enable_consistency = st.checkbox("Consistency checks", value=True)
        enable_skip_patterns = st.checkbox("Skip pattern validation", value=True)
        
        # Custom validation rules
        st.markdown("**Custom Rules**")
        custom_rules = st.text_area(
            "Add custom validation rules (one per line)",
            placeholder="age >= 0\nincome > 0\nage <= 120",
            help="Enter validation rules using column names and operators"
        )
    
    # Execute cleaning
    if st.button("🚀 Start Cleaning Process", type="primary"):
        with st.spinner("Processing data..."):
            try:
                # Configure cleaning parameters
                config = {
                    'missing_method': missing_method,
                    'outlier_methods': outlier_methods,
                    'enable_consistency': enable_consistency,
                    'enable_skip_patterns': enable_skip_patterns
                }
                
                if missing_method == "KNN":
                    config['knn_neighbors'] = knn_neighbors
                if "Z-score" in outlier_methods:
                    config['z_threshold'] = z_threshold
                if "Winsorization" in outlier_methods:
                    config['winsor_limits'] = winsor_limits
                if custom_rules.strip():
                    config['custom_rules'] = custom_rules.strip().split('\n')
                
                # Perform cleaning
                cleaned_data, cleaning_report = cleaner.clean_data(st.session_state.data, config)
                st.session_state.cleaned_data = cleaned_data
                
                # Log the process
                st.session_state.processing_log.append({
                    'step': 'Data Cleaning',
                    'timestamp': pd.Timestamp.now(),
                    'config': config,
                    'report': cleaning_report
                })
                
                st.success("✅ Data cleaning completed!")
                
                # Display cleaning results
                st.subheader("Cleaning Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows Before", f"{st.session_state.data.shape[0]:,}")
                with col2:
                    st.metric("Rows After", f"{cleaned_data.shape[0]:,}")
                with col3:
                    rows_removed = st.session_state.data.shape[0] - cleaned_data.shape[0]
                    st.metric("Rows Removed", f"{rows_removed:,}")
                
                # Detailed cleaning report
                st.subheader("Detailed Report")
                st.json(cleaning_report)
                
                # Preview cleaned data
                st.subheader("Cleaned Data Preview")
                st.dataframe(cleaned_data.head(10), width="stretch")
                
            except Exception as e:
                st.error(f"❌ Error during cleaning: {str(e)}")

def weight_application_page():
    st.header("⚖️ Weight Application & Statistical Computation")
    
    if st.session_state.cleaned_data is None:
        st.warning("⚠️ Please complete data cleaning first.")
        return
    
    calculator = WeightCalculator()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Weight Configuration")
        
        # Weight variable selection
        weight_column = st.selectbox(
            "Select Weight Variable",
            ["None"] + list(st.session_state.cleaned_data.columns),
            help="Choose the column containing survey weights"
        )
        
        trim_weights = False  # Default values
        trim_threshold = 97
        normalize_weights = True
        
        if weight_column != "None":
            # Weight diagnostics
            weights = st.session_state.cleaned_data[weight_column]
            
            # Convert to numeric if it's not already
            try:
                weights = pd.to_numeric(weights, errors='coerce')
            except:
                st.error("❌ Selected weight column contains non-numeric values")
                return
            
            if weights.isnull().all():
                st.error("❌ Weight column contains no valid numeric values")
                return
            
            st.markdown("**Weight Diagnostics**")
            col_w1, col_w2 = st.columns(2)
            
            with col_w1:
                st.metric("Min Weight", f"{weights.min():.3f}")
                st.metric("Mean Weight", f"{weights.mean():.3f}")
            
            with col_w2:
                st.metric("Max Weight", f"{weights.max():.3f}")
                st.metric("Std Weight", f"{weights.std():.3f}")
            
            # Weight adjustments
            st.markdown("**Weight Adjustments**")
            trim_weights = st.checkbox("Trim extreme weights")
            if trim_weights:
                trim_threshold = st.slider("Trim threshold (percentile)", 95, 99, 97)
            
            normalize_weights = st.checkbox("Normalize weights", value=True)
    
    with col2:
        st.subheader("Analysis Variables")
        
        # Select variables for analysis
        numeric_columns = st.session_state.cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = st.session_state.cleaned_data.select_dtypes(include=['object']).columns.tolist()
        
        analysis_vars = st.multiselect(
            "Variables for Analysis",
            numeric_columns + categorical_columns,
            help="Select variables to include in weighted analysis"
        )
        
        # Stratification variables
        strat_vars = st.multiselect(
            "Stratification Variables",
            categorical_columns,
            help="Variables for stratified analysis"
        )
        
        # Confidence level
        confidence_level = st.slider(
            "Confidence Level (%)",
            90, 99, 95,
            help="Confidence level for margin of error calculations"
        )
    
    # Execute weight application
    if st.button("📊 Apply Weights & Calculate Statistics", type="primary"):
        if not analysis_vars:
            st.error("❌ Please select at least one analysis variable.")
            return
        
        with st.spinner("Calculating weighted statistics..."):
            try:
                config = {
                    'weight_column': weight_column if weight_column != "None" else None,
                    'analysis_vars': analysis_vars,
                    'strat_vars': strat_vars,
                    'confidence_level': confidence_level / 100,
                    'trim_weights': trim_weights if weight_column != "None" else False,
                    'normalize_weights': normalize_weights if weight_column != "None" else False
                }
                
                if trim_weights and weight_column != "None":
                    config['trim_threshold'] = trim_threshold / 100
                
                # Calculate weighted results
                results = calculator.calculate_weighted_statistics(st.session_state.cleaned_data, config)
                st.session_state.weighted_results = results
                
                # Log the process
                st.session_state.processing_log.append({
                    'step': 'Weight Application',
                    'timestamp': pd.Timestamp.now(),
                    'config': config,
                    'summary': f"Analyzed {len(analysis_vars)} variables"
                })
                
                st.success("✅ Weighted statistics calculated!")
                
                # Display results
                st.subheader("Statistical Results")
                
                # Summary statistics
                if 'summary_stats' in results:
                    st.markdown("**Summary Statistics**")
                    st.dataframe(results['summary_stats'], width="stretch")
                
                # Margins of error
                if 'margins_of_error' in results:
                    st.markdown("**Margins of Error**")
                    st.dataframe(results['margins_of_error'], width="stretch")
                
                # Stratified results
                if 'stratified_results' in results and strat_vars:
                    st.markdown("**Stratified Analysis**")
                    for var in strat_vars:
                        if var in results['stratified_results']:
                            st.markdown(f"*By {var}:*")
                            st.dataframe(results['stratified_results'][var], width="stretch")
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.error(f"❌ Error calculating statistics: {str(e)}")
                with st.expander("Error Details (for debugging)"):
                    st.code(error_details)

def analysis_page():
    st.header("📈 Analysis & Visualization")
    
    if st.session_state.weighted_results is None:
        st.warning("⚠️ Please complete weight application first.")
        return
    
    viz = Visualizations()
    
    # Visualization options
    st.subheader("Data Quality Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Missing Data Patterns"):
            fig = viz.plot_missing_patterns(st.session_state.cleaned_data)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if st.button("📈 Distribution Analysis"):
            numeric_cols = st.session_state.cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:3]:  # Show first 3 numeric columns
                fig = viz.plot_distribution(st.session_state.cleaned_data, col)
                st.plotly_chart(fig, use_container_width=True)
    
    # Results visualization
    st.subheader("Statistical Results Visualization")
    
    if 'summary_stats' in st.session_state.weighted_results:
        # Interactive results table
        st.markdown("**Interactive Results Table**")
        results_df = st.session_state.weighted_results['summary_stats']
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            variable_filter = st.multiselect(
                "Filter Variables",
                results_df.index.tolist() if hasattr(results_df, 'index') else [],
                default=results_df.index.tolist()[:5] if hasattr(results_df, 'index') else []
            )
        
        with col2:
            stat_filter = st.multiselect(
                "Filter Statistics",
                results_df.columns.tolist() if hasattr(results_df, 'columns') else [],
                default=results_df.columns.tolist() if hasattr(results_df, 'columns') else []
            )
        
        # Filtered results
        if variable_filter and stat_filter:
            filtered_results = results_df.loc[variable_filter, stat_filter]
            st.dataframe(filtered_results, width="stretch")
            
            # Export filtered results
            csv = filtered_results.to_csv()
            st.download_button(
                label="📥 Download Filtered Results",
                data=csv,
                file_name="survey_results.csv",
                mime="text/csv"
            )

def report_generation_page():
    st.header("📄 Report Generation")
    
    if st.session_state.weighted_results is None:
        st.warning("⚠️ Please complete the analysis first.")
        return
    
    generator = ReportGenerator()
    
    # Report configuration
    st.subheader("Report Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_title = st.text_input("Report Title", "Survey Data Analysis Report")
        report_subtitle = st.text_input("Subtitle", "Statistical Agency Report")
        
        include_methodology = st.checkbox("Include Methodology Section", value=True)
        include_diagnostics = st.checkbox("Include Data Quality Diagnostics", value=True)
        include_visualizations = st.checkbox("Include Visualizations", value=True)
    
    with col2:
        report_format = st.selectbox("Report Format", ["PDF", "HTML"])
        
        # Template selection
        template_style = st.selectbox(
            "Report Template",
            ["Standard", "Executive Summary", "Technical Report"],
            help="Choose the report style and level of detail"
        )
        
        author_name = st.text_input("Author", "Statistical Agency")
        organization = st.text_input("Organization", "National Statistical Office")
    
    # Custom sections
    st.subheader("Custom Sections")
    custom_sections = st.text_area(
        "Additional Sections (one per line)",
        placeholder="Executive Summary\nKey Findings\nRecommendations",
        help="Add custom sections to include in the report"
    )
    
    # Generate report
    if st.button("🎯 Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            try:
                config = {
                    'title': report_title,
                    'subtitle': report_subtitle,
                    'author': author_name,
                    'organization': organization,
                    'format': report_format,
                    'template_style': template_style,
                    'include_methodology': include_methodology,
                    'include_diagnostics': include_diagnostics,
                    'include_visualizations': include_visualizations,
                    'custom_sections': custom_sections.strip().split('\n') if custom_sections.strip() else []
                }
                
                # Generate report
                report_data = generator.generate_report(
                    st.session_state.cleaned_data,
                    st.session_state.weighted_results,
                    st.session_state.processing_log,
                    config
                )
                
                st.success("✅ Report generated successfully!")
                
                # Display report preview
                st.subheader("Report Preview")
                st.markdown(report_data['preview'])
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if report_format == "PDF" and 'pdf_content' in report_data:
                        pdf_size = len(report_data['pdf_content'])
                        st.download_button(
                            label=f"📄 Download PDF Report ({pdf_size:,} bytes)",
                            data=report_data['pdf_content'],
                            file_name=f"survey_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )
                    elif report_format == "HTML" and 'html_content' in report_data:
                        st.download_button(
                            label="📄 Download HTML Report",
                            data=report_data['html_content'],
                            file_name=f"survey_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.html",
                            mime="text/html"
                        )
                
                with col2:
                    st.download_button(
                        label="💾 Download Data",
                        data=st.session_state.cleaned_data.to_csv(index=False),
                        file_name=f"processed_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.error(f"❌ Error generating report: {str(e)}")
                with st.expander("Error Details (for debugging)"):
                    st.code(error_details)

def processing_log_page():
    st.header("📋 Processing Log & Audit Trail")
    
    if not st.session_state.processing_log:
        st.info("ℹ️ No processing steps completed yet.")
        return
    
    # Display processing log
    st.subheader("Processing Steps")
    
    for i, log_entry in enumerate(st.session_state.processing_log):
        with st.expander(f"Step {i+1}: {log_entry['step']} - {log_entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
            st.json(log_entry)
    
    # Summary statistics
    st.subheader("Processing Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Steps", len(st.session_state.processing_log))
    
    with col2:
        if st.session_state.processing_log:
            duration = st.session_state.processing_log[-1]['timestamp'] - st.session_state.processing_log[0]['timestamp']
            st.metric("Total Duration", str(duration).split('.')[0])
    
    with col3:
        if st.session_state.data is not None:
            st.metric("Data Points Processed", f"{st.session_state.data.size:,}")
    
    # Export log
    if st.button("📥 Export Processing Log"):
        log_df = pd.DataFrame(st.session_state.processing_log)
        csv = log_df.to_csv(index=False)
        st.download_button(
            label="Download Log CSV",
            data=csv,
            file_name=f"processing_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    # Clear log option
    if st.button("🗑️ Clear Processing Log", type="secondary"):
        st.session_state.processing_log = []
        st.rerun()

if __name__ == "__main__":
    main()

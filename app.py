import streamlit as st
import pandas as pd
import numpy as np
from modules.data_loader import DataLoader
from modules.data_cleaner import DataCleaner
from modules.weight_calculator import WeightCalculator
from modules.report_generator import ReportGenerator
from modules.visualizations import Visualizations
from modules.schema_validator import SchemaValidator
from modules.audit_trail import AuditTrail
from modules.dashboard import Dashboard
from modules.user_guidance import UserGuidance
from modules.process_intelligence_chatbot import ProcessIntelligenceChatbot
import json

# Page configuration
st.set_page_config(
    page_title="Survey Data Processing Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better UI
st.markdown("""
<style>
    .stAlert > div {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .chat-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #fafafa;
    }
    .stSelectbox > label {
        color: #1e88e5;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

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
    
    # Show interactive tutorial for new users (if not completed)
    UserGuidance.show_interactive_tutorial()
    
    # Initialize audit trail
    audit = AuditTrail()
    
    # Sidebar navigation with enhanced options
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Select Processing Step",
        [
            "📁 Data Upload & Schema",
            "🧹 Data Cleaning",
            "⚖️ Weight Application", 
            "📈 Analysis & Visualization",
            "📄 Report Generation",
            "🔍 Audit Trail",
            "📋 Processing Log"
        ]
    )
    
    # Quick stats in sidebar
    if st.session_state.get('data') is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Quick Stats**")
        data = st.session_state.data
        st.sidebar.metric("Records", f"{data.shape[0]:,}")
        st.sidebar.metric("Columns", data.shape[1])
        missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100)
        st.sidebar.metric("Missing %", f"{missing_pct:.1f}%")
        
        # Processing progress
        st.sidebar.markdown("**🔄 Progress**")
        progress_items = [
            ("Upload", st.session_state.get('data') is not None),
            ("Schema", st.session_state.get('schema') is not None),
            ("Cleaning", st.session_state.get('cleaned_data') is not None),
            ("Weighting", st.session_state.get('weighted_results') is not None)
        ]
        completed = sum(1 for _, done in progress_items if done)
        st.sidebar.progress(completed / len(progress_items))
        st.sidebar.write(f"{completed}/{len(progress_items)} steps completed")
    
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
    
    # Add project tracking to sidebar
    display_project_tracking_sidebar()
    
    # Route to selected page
    if page == "📁 Data Upload & Schema":
        data_upload_page(audit)
    elif page == "🧹 Data Cleaning":
        data_cleaning_page(audit)
    elif page == "⚖️ Weight Application":
        weight_application_page(audit)
    elif page == "📈 Analysis & Visualization":
        analysis_page(audit)
    elif page == "📄 Report Generation":
        report_generation_page(audit)
    elif page == "🔍 Audit Trail":
        audit.display_audit_trail()
    elif page == "📋 Processing Log":
        processing_log_page()
    
    # Add the Process Intelligence Chatbot to every page
    if st.session_state.get('data') is not None or st.session_state.get('processing_log'):
        chatbot = ProcessIntelligenceChatbot(current_page=page)
        chatbot.display_chatbot()

def data_upload_page(audit):
    st.header("📁 Data Upload & Schema Configuration")
    
    # Show processing tips
    UserGuidance.show_processing_tips()
    
    # Show data quality alerts if data exists
    if st.session_state.get('data') is not None:
        UserGuidance.show_data_quality_alerts(st.session_state.data)
    
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
                
                # Log the upload action
                audit.log_data_upload(
                    filename=uploaded_file.name,
                    file_size=uploaded_file.size,
                    file_type=uploaded_file.type,
                    data_shape=data.shape if data is not None and hasattr(data, 'shape') else (0, 0)
                )
                
                shape_info = data.shape if data is not None and hasattr(data, 'shape') else "Unknown"
                st.success(f"✅ File loaded successfully! Shape: {shape_info}")
                st.info(f"📝 Upload logged to audit trail at {pd.Timestamp.now().strftime('%H:%M:%S')}")
                
                # Data preview
                st.subheader("Data Preview")
                if data is not None and not data.empty:
                    st.dataframe(data.head(10), width="stretch")
                else:
                    st.warning("No data to preview")
                
                # Basic statistics
                st.subheader("Basic Statistics")
                col_stats1, col_stats2 = st.columns(2)
                
                with col_stats1:
                    if data is not None:
                        st.metric("Rows", f"{data.shape[0]:,}")
                        st.metric("Numeric Columns", len(data.select_dtypes(include=[np.number]).columns))
                    else:
                        st.metric("Rows", "0")
                        st.metric("Numeric Columns", "0")
                
                with col_stats2:
                    if data is not None:
                        st.metric("Columns", data.shape[1])
                        st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
                    else:
                        st.metric("Columns", "0")
                        st.metric("Missing Values", "0")
                
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
                    
                    # Log schema detection
                    audit.log_schema_detection("auto-detected", schema)
                    
                    st.success("✅ Schema auto-detected!")
                    st.info("📝 Schema detection logged to audit trail")
                    
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

def data_cleaning_page(audit):
    st.header("🧹 Data Cleaning & Validation")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first in the Data Upload page.")
        return
    
    # Data cleaning interface
    
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
                
                # Log to audit trail
                audit.log_data_cleaning(
                    config=config,
                    before_shape=st.session_state.data.shape,
                    after_shape=cleaned_data.shape,
                    cleaning_report=cleaning_report
                )
                
                # Log the process
                st.session_state.processing_log.append({
                    'step': 'Data Cleaning',
                    'timestamp': pd.Timestamp.now(),
                    'config': config,
                    'report': cleaning_report
                })
                
                st.success("✅ Data cleaning completed!")
                st.info("📝 Cleaning process logged to audit trail")
                
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

def weight_application_page(audit):
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
                if weights.isnull().all():
                    st.error("❌ Weight column contains no valid numeric values")
                    return
            except Exception as e:
                st.error(f"❌ Error processing weight column: {str(e)}")
                return
            
            st.markdown("**Weight Diagnostics**")
            col_w1, col_w2 = st.columns(2)
            
            try:
                with col_w1:
                    st.metric("Min Weight", f"{float(weights.min()):.3f}")
                    st.metric("Mean Weight", f"{float(weights.mean()):.3f}")
                
                with col_w2:
                    st.metric("Max Weight", f"{float(weights.max()):.3f}")
                    st.metric("Std Weight", f"{float(weights.std()):.3f}")
            except Exception as e:
                st.warning(f"Could not calculate weight statistics: {str(e)}")
                st.info("Weight diagnostics temporarily unavailable")
            
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

def analysis_page(audit):
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

def report_generation_page(audit):
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

def detect_current_processing_step():
    """Detect current processing step for step-aware chatbot."""
    if st.session_state.get('weighted_results'):
        return 'analysis'
    elif st.session_state.get('cleaned_data') is not None:
        return 'weighting'
    elif st.session_state.get('schema'):
        return 'cleaning'
    elif st.session_state.get('data') is not None:
        return 'schema'
    else:
        return 'upload'

def display_project_tracking_sidebar():
    """Display comprehensive project tracking in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Project Tracking**")
    
    # Processing steps status
    steps = [
        ("📁 Data Upload", st.session_state.get('data') is not None),
        ("🗂️ Schema Config", st.session_state.get('schema') is not None),
        ("🧹 Data Cleaning", st.session_state.get('cleaned_data') is not None),
        ("⚖️ Weight Application", st.session_state.get('weighted_results') is not None),
        ("📄 Report Ready", False)  # Would be based on report generation
    ]
    
    completed = sum(1 for _, done in steps if done)
    progress = completed / len(steps)
    
    st.sidebar.progress(progress)
    st.sidebar.write(f"**Progress: {completed}/{len(steps)} completed**")
    
    # Step status indicators
    for step_name, completed_status in steps:
        if completed_status:
            st.sidebar.success(f"✅ {step_name}")
        else:
            st.sidebar.info(f"⏳ {step_name}")
    
    # Quick stats if data available
    if st.session_state.get('data') is not None:
        data = st.session_state.data
        st.sidebar.markdown("**📈 Quick Stats**")
        st.sidebar.metric("Records", f"{data.shape[0]:,}")
        st.sidebar.metric("Columns", data.shape[1])
        
        if st.session_state.get('cleaned_data') is not None:
            cleaned = st.session_state.cleaned_data
            rows_removed = data.shape[0] - cleaned.shape[0]
            st.sidebar.metric("Rows Cleaned", f"{rows_removed:,}")
    
    # Processing time tracking
    if st.session_state.get('processing_log'):
        st.sidebar.markdown("**⏱️ Recent Activity**")
        for entry in st.session_state.processing_log[-3:]:  # Last 3 entries
            timestamp = entry.get('timestamp', pd.Timestamp.now()).strftime('%H:%M')
            step = entry.get('step', 'Unknown')
            st.sidebar.write(f"• {timestamp}: {step}")

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

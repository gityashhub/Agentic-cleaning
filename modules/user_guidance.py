import streamlit as st
import pandas as pd

class UserGuidance:
    """Enhanced user guidance system with tooltips, inline explanations, and error-checking alerts."""
    
    @staticmethod
    def show_processing_tips():
        """Display processing tips and best practices."""
        with st.expander("💡 Pro Tips for Better Results"):
            st.markdown("""
            **🎯 Data Upload Tips:**
            - Ensure your CSV uses consistent delimiters
            - Check for special characters in column names
            - Verify date formats are consistent
            
            **🧹 Data Cleaning Best Practices:**
            - Review missing data patterns before imputation
            - Use KNN imputation for better accuracy with numeric data
            - Always validate results after cleaning
            
            **⚖️ Survey Weighting Guidelines:**
            - Check weight distribution before applying
            - Consider trimming extreme weights
            - Verify population targets match your sample
            
            **📊 Analysis Recommendations:**
            - Start with basic descriptive statistics
            - Check for outliers in key variables
            - Use stratified analysis for subgroup insights
            """)
    
    @staticmethod
    def show_data_quality_alerts(data):
        """Show data quality alerts and warnings."""
        if data is None:
            return
        
        alerts = []
        
        # Check for high missing data
        missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100)
        if missing_pct > 20:
            alerts.append(("❌ High Missing Data", f"Your dataset has {missing_pct:.1f}% missing values. Consider data imputation."))
        elif missing_pct > 10:
            alerts.append(("⚠️ Moderate Missing Data", f"Your dataset has {missing_pct:.1f}% missing values. Review before analysis."))
        
        # Check for duplicates
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            alerts.append(("⚠️ Duplicate Records", f"Found {duplicates} duplicate rows. Consider deduplication."))
        
        # Check for potential data type issues
        for col in data.columns[:5]:  # Check first 5 columns
            if data[col].dtype == 'object':
                try:
                    numeric_converted = pd.to_numeric(data[col], errors='coerce')
                    numeric_count = numeric_converted.count()  # Count non-null values
                    if numeric_count > len(data) * 0.8:
                        alerts.append(("💡 Type Conversion Suggestion", f"Column '{col}' appears to be numeric but stored as text."))
                except:
                    pass
        
        # Display alerts
        if alerts:
            st.markdown("### 🚨 Data Quality Alerts")
            for alert_type, message in alerts:
                if "❌" in alert_type:
                    st.error(f"{alert_type}: {message}")
                elif "⚠️" in alert_type:
                    st.warning(f"{alert_type}: {message}")
                else:
                    st.info(f"{alert_type}: {message}")
        else:
            st.success("✅ No major data quality issues detected!")
    
    @staticmethod
    def show_progress_indicator(current_step, total_steps, step_name):
        """Show progress indicator with current step."""
        progress = current_step / total_steps
        st.progress(progress)
        st.write(f"**Step {current_step}/{total_steps}: {step_name}**")
        
        if progress < 1.0:
            st.info(f"🎯 Next: Complete {step_name} to proceed to the next step.")
    
    @staticmethod
    def show_help_tooltip(help_text, key=None):
        """Show help tooltip with detailed explanation."""
        return st.help(help_text)
    
    @staticmethod
    def show_validation_feedback(validation_results):
        """Show validation feedback with specific errors and suggestions."""
        if not validation_results:
            return
        
        st.markdown("### 🔍 Validation Results")
        
        for result in validation_results:
            severity = result.get('severity', 'info')
            message = result.get('message', '')
            suggestion = result.get('suggestion', '')
            
            if severity == 'error':
                st.error(f"❌ {message}")
                if suggestion:
                    st.markdown(f"**💡 Suggestion:** {suggestion}")
            elif severity == 'warning':
                st.warning(f"⚠️ {message}")
                if suggestion:
                    st.markdown(f"**💡 Suggestion:** {suggestion}")
            else:
                st.info(f"ℹ️ {message}")
    
    @staticmethod
    def show_interactive_tutorial():
        """Show interactive tutorial for first-time users."""
        if 'tutorial_completed' not in st.session_state:
            st.session_state.tutorial_completed = False
        
        if not st.session_state.tutorial_completed:
            with st.expander("🎓 Interactive Tutorial - Click to Start!", expanded=True):
                st.markdown("""
                ### Welcome to the Survey Data Processing Platform! 🎉
                
                This platform helps statistical agencies process survey data with AI-powered tools.
                
                **🚀 Quick Start Guide:**
                
                1. **📁 Upload Your Data**
                   - Go to "Data Upload & Schema"
                   - Upload your CSV or Excel file
                   - Review the data preview and statistics
                
                2. **🧹 Clean Your Data**
                   - Navigate to "Data Cleaning"
                   - Configure cleaning parameters
                   - Apply imputation and outlier detection
                
                3. **⚖️ Apply Survey Weights**
                   - Go to "Weight Application"
                   - Select your weight variable
                   - Configure analysis variables
                
                4. **📊 Analyze Results**
                   - View the "Analytics Dashboard"
                   - Explore visualizations
                   - Review quality metrics
                
                5. **📄 Generate Reports**
                   - Create professional reports
                   - Export results and data
                
                6. **🔍 Track Everything**
                   - Monitor the "Audit Trail"
                   - Review processing history
                   - Export audit logs
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Got it! Start Processing"):
                        st.session_state.tutorial_completed = True
                        st.success("🎉 Tutorial completed! You can now start processing your data.")
                        st.rerun()
                
                with col2:
                    if st.button("📚 Show Tutorial Later"):
                        st.session_state.tutorial_completed = True
                        st.info("Tutorial hidden. You can always refer to the Help & Guidance in the sidebar.")
                        st.rerun()
    
    # Smart suggestions feature removed as requested
    
    @staticmethod
    def show_error_recovery_help(error_type, error_message):
        """Show error recovery suggestions."""
        st.error(f"❌ {error_message}")
        
        recovery_tips = {
            'file_format': [
                "✅ Ensure your file is in CSV or Excel format",
                "🔍 Check that the file isn't corrupted",
                "📝 Verify the file encoding (UTF-8 recommended)"
            ],
            'missing_data': [
                "🧹 Use the data cleaning tools to handle missing values",
                "📊 Consider removing columns with >90% missing data",
                "🎯 Apply appropriate imputation methods"
            ],
            'validation': [
                "🔍 Review your data for inconsistencies",
                "📝 Check variable types and formats",
                "⚙️ Adjust validation rules if needed"
            ]
        }
        
        if error_type in recovery_tips:
            st.markdown("**🛠️ Recovery Suggestions:**")
            for tip in recovery_tips[error_type]:
                st.info(tip)
    
    @staticmethod
    def add_contextual_help():
        """Add contextual help throughout the interface."""
        if st.session_state.get('current_page'):
            page = st.session_state.current_page
            
            help_content = {
                'dashboard': "The Analytics Dashboard provides real-time insights into your data processing workflow.",
                'upload': "Upload your survey data files and configure the data schema for processing.",
                'cleaning': "Clean and validate your data using advanced statistical methods.",
                'weighting': "Apply survey weights and calculate statistical estimates with confidence intervals.",
                'analysis': "Explore your data with interactive visualizations and quality assessments.",
                'reporting': "Generate professional reports documenting your methodology and findings.",
                'audit': "Review the complete audit trail of all processing steps and user actions."
            }
            
            if page in help_content:
                st.sidebar.markdown(f"**❓ About This Page**")
                st.sidebar.info(help_content[page])
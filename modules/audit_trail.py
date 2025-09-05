import pandas as pd
import streamlit as st
from datetime import datetime
import json
import uuid

class AuditTrail:
    """Comprehensive audit trail system for tracking all user actions and data transformations."""
    
    def __init__(self):
        if 'audit_log' not in st.session_state:
            st.session_state.audit_log = []
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
    
    def log_action(self, action_type, details, data_info=None, user_input=None):
        """Log a user action with comprehensive details."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': st.session_state.session_id,
            'action_type': action_type,
            'details': details,
            'data_info': data_info or {},
            'user_input': user_input or {},
            'entry_id': str(uuid.uuid4())
        }
        
        st.session_state.audit_log.append(log_entry)
        return log_entry['entry_id']
    
    def log_data_upload(self, filename, file_size, file_type, data_shape):
        """Log data upload action."""
        return self.log_action(
            action_type="DATA_UPLOAD",
            details=f"Uploaded file: {filename}",
            data_info={
                'filename': filename,
                'file_size': file_size,
                'file_type': file_type,
                'rows': data_shape[0],
                'columns': data_shape[1]
            }
        )
    
    def log_data_cleaning(self, config, before_shape, after_shape, cleaning_report):
        """Log data cleaning action."""
        return self.log_action(
            action_type="DATA_CLEANING",
            details=f"Applied data cleaning: {config.get('missing_method', 'None')} imputation",
            data_info={
                'before_shape': before_shape,
                'after_shape': after_shape,
                'rows_removed': before_shape[0] - after_shape[0],
                'cleaning_report': cleaning_report
            },
            user_input=config
        )
    
    def log_weight_application(self, config, results_summary):
        """Log weight application action."""
        return self.log_action(
            action_type="WEIGHT_APPLICATION",
            details=f"Applied weights to {len(config.get('analysis_vars', []))} variables",
            data_info=results_summary,
            user_input=config
        )
    
    def log_report_generation(self, config, report_type):
        """Log report generation action."""
        return self.log_action(
            action_type="REPORT_GENERATION",
            details=f"Generated {report_type} report: {config.get('title', 'Untitled')}",
            user_input=config
        )
    
    def log_schema_detection(self, schema_type, schema_data):
        """Log schema detection/configuration."""
        return self.log_action(
            action_type="SCHEMA_CONFIGURATION",
            details=f"Schema {schema_type}",
            data_info={'schema': schema_data}
        )
    
    def get_audit_log_df(self):
        """Return audit log as a pandas DataFrame."""
        if not st.session_state.audit_log:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(st.session_state.audit_log)
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            # Return empty DataFrame with proper structure on error
            return pd.DataFrame(columns=['timestamp', 'action_type', 'details', 'entry_id', 'session_id'])
    
    def get_session_summary(self):
        """Get summary statistics for the current session."""
        if not st.session_state.audit_log:
            return {}
        
        df = self.get_audit_log_df()
        
        return {
            'total_actions': len(df),
            'session_duration': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60,
            'actions_by_type': df['action_type'].value_counts().to_dict(),
            'session_start': df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
            'last_activity': df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def export_audit_log(self, format='json'):
        """Export audit log in specified format."""
        if format == 'json':
            return json.dumps(st.session_state.audit_log, indent=2, default=str)
        elif format == 'csv':
            df = self.get_audit_log_df()
            return df.to_csv(index=False)
        else:
            raise ValueError("Format must be 'json' or 'csv'")
    
    def display_audit_trail(self):
        """Display interactive audit trail interface."""
        st.header("🔍 Audit Trail & Processing History")
        
        if not st.session_state.audit_log:
            st.info("📝 No actions recorded yet. Start processing data to see audit trail.")
            return
        
        # Session summary
        summary = self.get_session_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Actions", summary['total_actions'])
        with col2:
            st.metric("Session Duration", f"{summary['session_duration']:.1f} min")
        with col3:
            st.metric("Session Start", summary['session_start'])
        with col4:
            st.metric("Last Activity", summary['last_activity'])
        
        # Action type breakdown
        st.subheader("📊 Action Breakdown")
        action_counts = summary['actions_by_type']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create a simple bar chart
            action_df = pd.DataFrame(list(action_counts.items()), columns=['Action Type', 'Count'])
            st.bar_chart(action_df.set_index('Action Type'))
        
        with col2:
            for action, count in action_counts.items():
                st.metric(action.replace('_', ' ').title(), count)
        
        # Detailed audit log
        st.subheader("📋 Detailed Audit Log")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            action_filter = st.multiselect(
                "Filter by Action Type",
                options=list(action_counts.keys()),
                default=list(action_counts.keys())
            )
        
        with col2:
            show_details = st.checkbox("Show Technical Details", value=False)
        
        # Display filtered log
        df = self.get_audit_log_df()
        
        if not df.empty:
            if action_filter:
                df = df[df['action_type'].isin(action_filter)]
            
            # Format for display
            if not df.empty and all(col in df.columns for col in ['timestamp', 'action_type', 'details']):
                display_df = df[['timestamp', 'action_type', 'details']].copy()
                if 'timestamp' in display_df.columns and hasattr(display_df['timestamp'].iloc[0], 'strftime'):
                    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
                else:
                    display_df['timestamp'] = display_df['timestamp'].astype(str)
                display_df.columns = ['Time', 'Action Type', 'Description']
            else:
                display_df = pd.DataFrame(columns=['Time', 'Action Type', 'Description'])
        else:
            display_df = pd.DataFrame(columns=['Time', 'Action Type', 'Description'])
        
        st.dataframe(display_df, use_container_width=True)
        
        # Export options
        st.subheader("📥 Export Audit Trail")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Download as JSON"):
                json_data = self.export_audit_log('json')
                st.download_button(
                    label="💾 Download JSON",
                    data=json_data,
                    file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Download as CSV"):
                csv_data = self.export_audit_log('csv')
                st.download_button(
                    label="💾 Download CSV",
                    data=csv_data,
                    file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Technical details expansion
        if show_details and not df.empty:
            st.subheader("🔧 Technical Details")
            for idx, row in df.iterrows():
                with st.expander(f"{row['timestamp'].strftime('%H:%M:%S')} - {row['action_type']}"):
                    st.json({
                        'Entry ID': row['entry_id'],
                        'Session ID': row['session_id'],
                        'Data Info': row['data_info'],
                        'User Input': row['user_input']
                    })
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from groq import Groq
import traceback

class ProcessIntelligenceChatbot:
    """Step-aware AI Assistant with proper chat interface and cross-step memory."""
    
    def __init__(self, current_step=None, current_page=None):
        # Current processing step awareness - now based on actual page selection
        self.current_page = current_page
        self.current_step = current_step or self._detect_current_step_from_page(current_page)
        
        # Initialize Groq client
        self.client = None
        self._init_groq_client()
        
        # Initialize step-aware chat history with proper isolation
        if 'step_chat_messages' not in st.session_state:
            st.session_state.step_chat_messages = {}
        
        # Initialize cross-step memory for remembering previous conversations
        if 'cross_step_memory' not in st.session_state:
            st.session_state.cross_step_memory = {}
            
        # Track current active step to detect transitions
        if 'current_active_step' not in st.session_state:
            st.session_state.current_active_step = None
        
        # Language settings
        if 'chatbot_language' not in st.session_state:
            st.session_state.chatbot_language = 'english'
        
        # Track first query per step
        if 'step_first_query' not in st.session_state:
            st.session_state.step_first_query = {}
        
        # Initialize current step with clean history
        self._initialize_step_chatbot()
            
        # Context cache with step awareness
        self._context_cache = {}
        self._step_summaries = {}
    
    def _initialize_step_chatbot(self):
        """Initialize chatbot for current step with proper isolation."""
        # Detect step transition
        if st.session_state.current_active_step != self.current_step:
            # Store previous step's conversation in cross-step memory before switching
            if st.session_state.current_active_step is not None:
                self._store_step_conversation_in_memory(st.session_state.current_active_step)
            
            # Update active step
            st.session_state.current_active_step = self.current_step
            
            # Reset first query flag for new step
            if self.current_step not in st.session_state.step_first_query:
                st.session_state.step_first_query[self.current_step] = False
        
        # Ensure current step has clean message history
        if self.current_step not in st.session_state.step_chat_messages:
            st.session_state.step_chat_messages[self.current_step] = []
    
    def _store_step_conversation_in_memory(self, step):
        """Store step's conversation in cross-step memory for future reference."""
        if step in st.session_state.step_chat_messages:
            messages = st.session_state.step_chat_messages[step]
            if messages:  # Only store if there are messages
                # Create a summary of the conversation
                conversation_summary = {
                    'step': step,
                    'message_count': len(messages),
                    'last_activity': messages[-1].get('timestamp', 'Unknown') if messages else None,
                    'key_topics': self._extract_key_topics(messages),
                    'full_conversation': messages  # Store full conversation for reference
                }
                st.session_state.cross_step_memory[step] = conversation_summary
    
    def _extract_key_topics(self, messages):
        """Extract key topics from conversation for quick reference."""
        topics = []
        for msg in messages[-5:]:  # Look at last 5 messages
            if msg['role'] == 'user':
                content = msg['content'].lower()
                # Simple keyword extraction
                if 'upload' in content or 'file' in content:
                    topics.append('data_upload')
                elif 'clean' in content or 'missing' in content:
                    topics.append('data_cleaning')
                elif 'weight' in content or 'survey' in content:
                    topics.append('weighting')
                elif 'analysis' in content or 'result' in content:
                    topics.append('analysis')
        return list(set(topics))  # Remove duplicates
    
    def _get_intelligent_response(self, user_message, current_step):
        """Generate intelligent responses based on user intent and real data."""
        message_lower = user_message.lower()
        context = self._get_step_context(current_step)
        
        # Handle complaints about repetition
        repetition_keywords = ['same answer', 'repeating', 'repeat', 'generic', 'template', 'boring']
        if any(keyword in message_lower for keyword in repetition_keywords):
            return "I apologize for the repetitive responses! Let me be more specific and helpful. What exact information would you like me to provide about your data or the current processing step?"
        
        # Handle specific data queries with real numbers
        data_info = context.get('data_info', {})
        detailed_stats = context.get('detailed_stats', {})
        processing_results = context.get('processing_results', {})
        
        # Missing values queries
        if any(keyword in message_lower for keyword in ['missing', 'null', 'empty', 'blank']):
            if data_info:
                missing_count = data_info.get('missing_values', 0)
                missing_pct = data_info.get('missing_percentage', 0)
                total_rows = data_info.get('total_rows', 0)
                columns_with_missing = detailed_stats.get('columns_with_missing', [])
                
                response = f"📊 **Missing Values Analysis:**\n"
                response += f"• Total missing values: **{missing_count:,}**\n"
                response += f"• Missing percentage: **{missing_pct:.2f}%**\n"
                response += f"• Total rows: **{total_rows:,}**\n\n"
                
                if columns_with_missing:
                    response += "**Columns with most missing values:**\n"
                    for col, count in columns_with_missing[:3]:
                        response += f"• {col}: {count} missing\n"
                
                return response
            return "I don't see any data uploaded yet. Please upload your data file first to analyze missing values."
        
        # Records/rows queries
        if any(keyword in message_lower for keyword in ['records', 'rows', 'entries', 'how many']):
            if data_info:
                rows = data_info.get('total_rows', 0)
                cols = data_info.get('total_columns', 0)
                response = f"📈 **Data Overview:**\n"
                response += f"• **{rows:,}** total records/rows\n"
                response += f"• **{cols}** columns\n"
                
                # Add cleaning info if available
                if 'cleaning' in processing_results:
                    cleaning = processing_results['cleaning']
                    rows_after = cleaning.get('rows_after_cleaning', rows)
                    removed = cleaning.get('rows_removed', 0)
                    if removed > 0:
                        response += f"• **{removed:,}** rows removed during cleaning\n"
                        response += f"• **{rows_after:,}** rows remaining after cleaning\n"
                        
                return response
            return "No data has been uploaded yet. Upload your CSV or Excel file to see record counts."
        
        # Columns queries
        if any(keyword in message_lower for keyword in ['columns', 'fields', 'variables']):
            if data_info and detailed_stats:
                total_cols = data_info.get('total_columns', 0)
                numeric_cols = data_info.get('numeric_columns', 0)
                categorical_cols = data_info.get('categorical_columns', 0)
                column_names = detailed_stats.get('column_names', [])
                
                response = f"📋 **Column Analysis:**\n"
                response += f"• **{total_cols}** total columns\n"
                response += f"• **{numeric_cols}** numeric columns\n"
                response += f"• **{categorical_cols}** categorical columns\n\n"
                
                if column_names:
                    response += f"**First {len(column_names)} columns:** {', '.join(column_names)}\n"
                    
                return response
            return "No data schema available. Please upload your data to see column information."
        
        # Cleaning process queries
        if any(keyword in message_lower for keyword in ['cleaning', 'clean', 'how', 'process', 'done', 'method']):
            if 'cleaning' in processing_results:
                cleaning = processing_results['cleaning']
                original_rows = data_info.get('total_rows', 0)
                remaining_rows = cleaning.get('rows_after_cleaning', 0)
                removed_rows = cleaning.get('rows_removed', 0)
                fixed_missing = cleaning.get('missing_values_fixed', 0)
                
                response = f"🧹 **Data Cleaning Summary:**\n\n"
                response += f"**Original Data:**\n• {original_rows:,} total rows\n• {data_info.get('missing_values', 0):,} missing values\n\n"
                response += f"**Cleaning Results:**\n• {remaining_rows:,} rows kept ({((remaining_rows/original_rows)*100):.1f}%)\n"
                if removed_rows > 0:
                    response += f"• {removed_rows:,} rows removed ({((removed_rows/original_rows)*100):.1f}%)\n"
                if fixed_missing > 0:
                    response += f"• {fixed_missing:,} missing values fixed\n"
                response += f"• Data quality: {cleaning.get('quality_improvement', 'improved')}\n\n"
                response += "**Methods Used:** Missing value imputation, outlier detection, duplicate removal"
                
                return response
            else:
                return f"Data cleaning hasn't been completed yet. Your current data has {data_info.get('missing_values', 0):,} missing values ({data_info.get('missing_percentage', 0):.2f}%) that can be addressed in the cleaning step."
        
        # Status and progress queries
        if any(keyword in message_lower for keyword in ['status', 'progress', 'completed', 'done']):
            step_status = context.get('step_status', 'pending')
            response = f"🔄 **Current Status: {current_step.title()} - {step_status.title()}**\n\n"
            
            if data_info:
                response += f"✅ Data uploaded: {data_info.get('total_rows', 0):,} rows, {data_info.get('total_columns', 0)} columns\n"
            
            if 'cleaning' in processing_results:
                cleaning = processing_results['cleaning']
                response += f"✅ Data cleaned: {cleaning.get('rows_after_cleaning', 0):,} rows remaining\n"
                
            if 'weighting' in processing_results:
                response += f"✅ Weights applied: Analysis ready\n"
                
            return response
            
        return None  # No intelligent response found, use AI
    
    def _get_recent_chat_history(self, current_step):
        """Get recent chat history for context."""
        if current_step not in st.session_state.step_chat_messages:
            return "No previous conversation."
            
        messages = st.session_state.step_chat_messages[current_step]
        if len(messages) <= 1:  # Only welcome message
            return "This is the start of our conversation."
            
        # Get last 4 messages (2 user + 2 assistant)
        recent = messages[-4:] if len(messages) >= 4 else messages[1:]  # Skip welcome message
        history_str = ""
        
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Assistant"
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            history_str += f"{role}: {content}\n"
            
        return history_str
    
    def _build_comprehensive_context(self, context, user_message):
        """Build detailed context string with all available real data and processing configurations for AI prompts."""
        context_parts = []
        
        # Current step and status
        context_parts.append(f"CURRENT STEP: {context['step'].upper()}")
        context_parts.append(f"STEP STATUS: {context.get('step_status', 'pending').upper()}")
        
        # Real data information
        data_info = context.get('data_info', {})
        if data_info:
            context_parts.append(f"\nREAL DATA METRICS:")
            context_parts.append(f"- Dataset: {data_info.get('total_rows', 0):,} rows × {data_info.get('total_columns', 0)} columns")
            context_parts.append(f"- Data quality: {data_info.get('missing_values', 0):,} missing values ({data_info.get('missing_percentage', 0):.2f}%)")
            context_parts.append(f"- Data types: {data_info.get('numeric_columns', 0)} numeric, {data_info.get('categorical_columns', 0)} categorical columns")
            if data_info.get('duplicates', 0) > 0:
                context_parts.append(f"- Duplicates found: {data_info.get('duplicates', 0):,}")
        
        # Real processing configurations used
        processing_configs = context.get('processing_configurations', {})
        if processing_configs:
            context_parts.append(f"\nACTUAL PROCESSING METHODS USED:")
            
            # Upload configuration
            if 'upload' in processing_configs:
                upload_config = processing_configs['upload']
                context_parts.append(f"UPLOAD STEP:")
                context_parts.append(f"- Data completeness: {upload_config.get('initial_quality_check', {}).get('data_completeness', 'Unknown')}%")
                context_parts.append(f"- Columns detected: {len(upload_config.get('columns_detected', []))}")
            
            # Cleaning configuration with actual methods
            if 'cleaning' in processing_configs:
                cleaning_config = processing_configs['cleaning']
                context_parts.append(f"CLEANING STEP METHODS:")
                context_parts.append(f"- Missing value method: {cleaning_config.get('imputation_method', 'None')}")
                if cleaning_config.get('outlier_detection'):
                    context_parts.append(f"- Outlier detection: {', '.join(cleaning_config['outlier_detection'])}")
                if cleaning_config.get('validation_rules'):
                    context_parts.append(f"- Validation rules: {', '.join(cleaning_config['validation_rules'])}")
                if cleaning_config.get('parameters'):
                    params = cleaning_config['parameters']
                    if params.get('knn_neighbors') and cleaning_config.get('imputation_method') == 'KNN':
                        context_parts.append(f"- KNN neighbors: {params['knn_neighbors']}")
                    if params.get('z_threshold') and 'Z-score' in cleaning_config.get('outlier_detection', []):
                        context_parts.append(f"- Z-score threshold: {params['z_threshold']}")
            
            # Weighting configuration with actual methods
            if 'weighting' in processing_configs:
                weighting_config = processing_configs['weighting']
                context_parts.append(f"WEIGHTING STEP METHODS:")
                context_parts.append(f"- Weight column: {weighting_config.get('weight_column', 'None')}")
                if weighting_config.get('analysis_variables'):
                    context_parts.append(f"- Analysis variables: {', '.join(weighting_config['analysis_variables'])}")
                if weighting_config.get('statistical_methods'):
                    context_parts.append(f"- Statistical methods: {', '.join(weighting_config['statistical_methods'])}")
                context_parts.append(f"- Confidence level: {weighting_config.get('confidence_level', 0.95)*100}%")
        
        # Processing results with actual outcomes
        processing_results = context.get('processing_results', {})
        if processing_results:
            context_parts.append(f"\nACTUAL PROCESSING OUTCOMES:")
            
            # Cleaning results
            if 'cleaning' in processing_results:
                cleaning = processing_results['cleaning']
                context_parts.append(f"CLEANING RESULTS:")
                context_parts.append(f"- Rows processed: {cleaning.get('rows_after_cleaning', 0):,} remaining")
                if cleaning.get('rows_removed', 0) > 0:
                    context_parts.append(f"- Rows removed: {cleaning['rows_removed']:,} ({cleaning.get('removal_percentage', 0):.1f}%)")
                context_parts.append(f"- Missing values fixed: {cleaning.get('missing_values_fixed', 0):,}")
                context_parts.append(f"- Quality improvement: {cleaning.get('quality_improvement', 'unknown')}")
            
            # Weighting results
            if 'weighting' in processing_results:
                context_parts.append(f"WEIGHTING RESULTS:")
                context_parts.append(f"- Statistical analysis completed")
                context_parts.append(f"- Survey weights applied successfully")
            
            # Recent processing activity
            if 'recent_activity' in processing_results:
                context_parts.append(f"RECENT ACTIVITY:")
                for activity in processing_results['recent_activity'][-3:]:  # Last 3 activities
                    context_parts.append(f"- {activity}")
        
        # Cross-step memory context
        if hasattr(st.session_state, 'cross_step_memory') and st.session_state.cross_step_memory:
            context_parts.append(f"\nCROSS-STEP MEMORY:")
            for step_name, memory in st.session_state.cross_step_memory.items():
                if memory.get('completed') and step_name != context['step']:
                    context_parts.append(f"- {step_name.upper()}: {memory.get('process_details', {})}")
        
        return "\n".join(context_parts)
    
    def _create_enhanced_system_prompt(self, language, current_step, context):
        """Create process-aware system prompt that includes real methodology details."""
        # Extract processing configurations for current and previous steps
        processing_configs = context.get('processing_configurations', {})
        processing_results = context.get('processing_results', {})
        data_info = context.get('data_info', {})
        
        # Build step-specific methodology context
        methodology_context = []
        
        # Upload methodology details
        if 'upload' in processing_configs:
            upload_config = processing_configs['upload']
            methodology_context.append(f"UPLOAD COMPLETED: Processed {len(upload_config.get('columns_detected', []))} columns with {upload_config.get('initial_quality_check', {}).get('data_completeness', 'unknown')}% data completeness")
        
        # Cleaning methodology details
        if 'cleaning' in processing_configs:
            cleaning_config = processing_configs['cleaning']
            cleaning_methods = []
            if cleaning_config.get('imputation_method') and cleaning_config['imputation_method'] != 'None':
                cleaning_methods.append(f"{cleaning_config['imputation_method']} imputation")
                if cleaning_config.get('parameters', {}).get('knn_neighbors') and cleaning_config['imputation_method'] == 'KNN':
                    cleaning_methods.append(f"(k={cleaning_config['parameters']['knn_neighbors']})")
            if cleaning_config.get('outlier_detection'):
                cleaning_methods.append(f"{', '.join(cleaning_config['outlier_detection'])} outlier detection")
                if 'Z-score' in cleaning_config['outlier_detection'] and cleaning_config.get('parameters', {}).get('z_threshold'):
                    cleaning_methods.append(f"(z-threshold={cleaning_config['parameters']['z_threshold']})")
            if cleaning_config.get('validation_rules'):
                cleaning_methods.append(f"validation rules: {', '.join(cleaning_config['validation_rules'])}")
                
            if cleaning_methods:
                methodology_context.append(f"CLEANING COMPLETED: Used {', '.join(cleaning_methods)}")
                
            # Add cleaning results
            if 'cleaning' in processing_results:
                cleaning_result = processing_results['cleaning']
                methodology_context.append(f"CLEANING OUTCOMES: {cleaning_result.get('rows_after_cleaning', 0)} rows retained, {cleaning_result.get('missing_values_fixed', 0)} missing values fixed")
        
        # Weighting methodology details  
        if 'weighting' in processing_configs:
            weighting_config = processing_configs['weighting']
            weighting_methods = []
            if weighting_config.get('weight_column') and weighting_config['weight_column'] != 'None':
                weighting_methods.append(f"weight column: {weighting_config['weight_column']}")
            if weighting_config.get('analysis_variables'):
                weighting_methods.append(f"analyzed {len(weighting_config['analysis_variables'])} variables")
            if weighting_config.get('statistical_methods'):
                weighting_methods.append(f"methods: {', '.join(weighting_config['statistical_methods'])}")
            weighting_methods.append(f"{weighting_config.get('confidence_level', 0.95)*100}% confidence level")
            
            if weighting_methods:
                methodology_context.append(f"WEIGHTING COMPLETED: Applied {', '.join(weighting_methods)}")
        
        # Current step context
        current_step_context = {
            'upload': f"You are helping with DATA UPLOAD. Dataset has {data_info.get('total_rows', 0):,} rows and {data_info.get('total_columns', 0)} columns.",
            'cleaning': f"You are helping with DATA CLEANING. Current data quality: {data_info.get('missing_percentage', 0):.1f}% missing values.",
            'weighting': f"You are helping with WEIGHT APPLICATION AND ANALYSIS. Ready to apply survey weights to cleaned data.",
            'analysis': f"You are helping with DATA ANALYSIS. All processing steps completed, ready for statistical analysis."
        }.get(current_step, f"You are helping with {current_step.upper()} processing.")
        
        # Build comprehensive system prompt
        prompt = f"""You are an intelligent Survey Data Processing Assistant with COMPLETE KNOWLEDGE of the actual processing methodology used.

{current_step_context}

ACTUAL PROCESSING METHODOLOGY COMPLETED:
{chr(10).join(methodology_context) if methodology_context else "No processing steps completed yet."}

REAL DATA CONTEXT:
- Dataset: {data_info.get('total_rows', 0):,} rows × {data_info.get('total_columns', 0)} columns
- Data quality: {data_info.get('missing_values', 0):,} missing values ({data_info.get('missing_percentage', 0):.2f}%)
- Data types: {data_info.get('numeric_columns', 0)} numeric, {data_info.get('categorical_columns', 0)} categorical
- Current step status: {context.get('step_status', 'pending').upper()}

KEY INSTRUCTIONS:
1. **PROCESS AWARENESS**: You know exactly what processing methods were used (imputation techniques, outlier detection methods, parameters, etc.)
2. **METHODOLOGY SPECIFIC**: Reference the actual methods used, not generic ones (e.g., "your KNN imputation with k=5" not just "missing value handling")
3. **PARAMETER AWARE**: Include specific parameters when discussing methods (thresholds, neighbor counts, confidence levels)
4. **OUTCOME FOCUSED**: Reference actual processing outcomes and results achieved
5. **STEP BRIDGING**: Connect information between completed steps using real methodology details
6. **DATA SPECIFIC**: Use exact numbers from this dataset, not examples

RESPONSE GUIDELINES:
- Always reference the actual methods and parameters used in processing
- Be specific about what was accomplished in each step
- Provide insights based on the real methodology applied to this data
- Connect processing choices to outcomes when explaining results
- Answer questions about methodology with specific implementation details

LANGUAGE: Respond in {language}.
TONE: Expert, specific, methodology-aware, helpful."""
        
        return prompt
    
    def _get_contextual_fallback(self, user_message, language, current_step):
        """Provide contextual fallback response with real data when possible."""
        context = self._get_step_context(current_step)
        data_info = context.get('data_info', {})
        
        # If we have data, provide real statistics
        if data_info:
            return f"I'm having trouble connecting to the AI service, but I can tell you about your data: You have {data_info.get('total_rows', 0):,} records with {data_info.get('total_columns', 0)} columns. Missing values: {data_info.get('missing_values', 0):,} ({data_info.get('missing_percentage', 0):.2f}%). What specific aspect would you like to know more about?"
        
        return f"I'm having trouble connecting to the AI service. Please upload your data first, then I can provide specific information about your dataset. What would you like to do in the {current_step} step?"
    
    def _get_relevant_cross_step_context(self, user_message):
        """Get relevant context from previous steps based on user's question."""
        if not st.session_state.cross_step_memory:
            return "No previous step conversations available."
        
        relevant_context = []
        user_msg_lower = user_message.lower()
        
        # Keywords that indicate questions about previous steps
        step_keywords = {
            'upload': ['upload', 'file', 'data loading', 'import'],
            'cleaning': ['clean', 'missing', 'duplicate', 'outlier', 'validation'],
            'weighting': ['weight', 'survey weight', 'statistical'],
            'analysis': ['analysis', 'result', 'visualization', 'report']
        }
        
        # Check if user is asking about specific previous steps
        for step, memory in st.session_state.cross_step_memory.items():
            step_mentioned = False
            
            # Check if step keywords match user question
            if step in step_keywords:
                for keyword in step_keywords[step]:
                    if keyword in user_msg_lower:
                        step_mentioned = True
                        break
            
            # Also check if step name is directly mentioned
            if step in user_msg_lower:
                step_mentioned = True
            
            if step_mentioned:
                # Get summary from memory
                relevant_context.append(f"""
                STEP {step.upper()}:
                - {memory.get('summary', 'No summary available')}
                - Messages: {memory.get('message_count', 0)}
                - Last activity: {memory.get('last_activity', 'Unknown')}
                - Topics discussed: {', '.join(memory.get('key_topics', []))}
                """)
        
        if relevant_context:
            return '\n'.join(relevant_context)
        else:
            # If no specific step mentioned, provide general overview
            overview = []
            for step, memory in st.session_state.cross_step_memory.items():
                overview.append(f"- {step}: {memory.get('summary', 'completed')}")
            
            if overview:
                return f"Previous steps completed:\n" + '\n'.join(overview)
            else:
                return "No previous step conversations available."

    def _detect_current_step_from_page(self, current_page=None):
        """Detect current processing step based on actual page selection."""
        if current_page is None:
            # Fallback to old logic if no page provided
            return self._detect_current_step_fallback()
        
        # Map page names to step names
        page_to_step = {
            "📁 Data Upload & Schema": "upload",
            "🧹 Data Cleaning": "cleaning", 
            "⚖️ Weight Application": "weighting",
            "📈 Analysis & Visualization": "analysis",
            "📄 Report Generation": "analysis",  # Report generation uses analysis context
            "🔍 Audit Trail": "analysis",  # Audit trail uses analysis context
            "📋 Processing Log": "analysis"  # Processing log uses analysis context
        }
        
        return page_to_step.get(current_page, "upload")
    
    def _detect_current_step_fallback(self):
        """Fallback step detection based on session state (old logic)."""
        if st.session_state.get('weighted_results'):
            return 'analysis'
        elif st.session_state.get('cleaned_data') is not None:
            return 'weighting'
        elif st.session_state.get('schema'):
            return 'cleaning'
        elif st.session_state.get('data') is not None:
            return 'upload'  # Changed from 'schema' to 'upload' to match page
        else:
            return 'upload'
    
    def _init_groq_client(self):
        """Initialize Groq client with better error handling."""
        try:
            api_key = os.getenv('GROQ_API_KEY')
            if api_key and api_key.strip():
                from groq import Groq
                self.client = Groq(api_key=api_key.strip())
                # Test the connection with a simple call
                test_response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10,
                    timeout=5
                )
                if test_response:
                    print("✅ GROQ client initialized successfully")
                else:
                    self.client = None
            else:
                print("❌ No GROQ_API_KEY found")
                self.client = None
        except Exception as e:
            print(f"❌ GROQ client initialization failed: {str(e)}")
            self.client = None

    def _get_step_context(self, step=None):
        """Get comprehensive context specific to a processing step with real configurations and processing details."""
        target_step = step or self.current_step
        
        # Don't cache context as data changes frequently
        context = {
            'step': target_step,
            'data_info': {},
            'step_status': 'pending',
            'detailed_stats': {},
            'processing_results': {},
            'processing_configurations': {},
            'actionable_insights': []
        }
        
        try:
            # === UPLOAD STEP CONTEXT ===
            if st.session_state.get('data') is not None:
                data = st.session_state.data
                if data is not None and hasattr(data, 'shape'):
                    total_cells = data.shape[0] * data.shape[1]
                    missing_count = data.isnull().sum().sum()
                    
                    # Basic data information
                    context['data_info'] = {
                        'total_rows': int(data.shape[0]),
                        'total_columns': int(data.shape[1]),
                        'missing_values': int(missing_count),
                        'missing_percentage': float((missing_count / total_cells) * 100) if total_cells > 0 else 0,
                        'numeric_columns': int(len(data.select_dtypes(include=[np.number]).columns)),
                        'categorical_columns': int(len(data.select_dtypes(include=['object']).columns)),
                        'duplicates': int(data.duplicated().sum()) if hasattr(data, 'duplicated') else 0
                    }
                    
                    # Detailed column analysis
                    missing_by_column = data.isnull().sum().sort_values(ascending=False)
                    context['detailed_stats'] = {
                        'columns_with_missing': list(missing_by_column[missing_by_column > 0].head(5).to_dict().items()),
                        'complete_columns': int((missing_by_column == 0).sum()),
                        'column_names': list(data.columns[:10]),
                        'data_types': data.dtypes.value_counts().to_dict()
                    }
                    
                    # Upload step configuration (file type, encoding, etc.)
                    context['processing_configurations']['upload'] = {
                        'file_type': 'detected_from_upload',
                        'columns_detected': list(data.columns),
                        'dtypes_inferred': data.dtypes.to_dict(),
                        'initial_quality_check': {
                            'has_missing': missing_count > 0,
                            'has_duplicates': context['data_info']['duplicates'] > 0,
                            'data_completeness': round((1 - context['data_info']['missing_percentage']/100) * 100, 1)
                        }
                    }
                    
                    if target_step == 'upload':
                        context['step_status'] = 'completed'
            
            # === SCHEMA STEP CONTEXT ===
            if st.session_state.get('schema') is not None:
                schema = st.session_state.schema
                context['processing_configurations']['schema'] = {
                    'schema_defined': True,
                    'configured_columns': schema if isinstance(schema, dict) else {},
                    'validation_rules_applied': bool(schema)
                }
                if target_step == 'schema':
                    context['step_status'] = 'completed'

            # === CLEANING STEP CONTEXT ===
            if st.session_state.get('cleaned_data') is not None:
                cleaned = st.session_state.cleaned_data
                if cleaned is not None and hasattr(cleaned, 'shape'):
                    original_rows = context['data_info'].get('total_rows', 0)
                    rows_removed = original_rows - cleaned.shape[0] if original_rows > 0 else 0
                    original_missing = context['data_info'].get('missing_values', 0)
                    remaining_missing = int(cleaned.isnull().sum().sum())
                    
                    # Enhanced cleaning results with actual process details
                    context['processing_results']['cleaning'] = {
                        'rows_after_cleaning': int(cleaned.shape[0]),
                        'rows_removed': int(rows_removed),
                        'removal_percentage': float((rows_removed / original_rows * 100)) if original_rows > 0 else 0,
                        'original_missing': original_missing,
                        'remaining_missing': remaining_missing,
                        'missing_values_fixed': original_missing - remaining_missing,
                        'quality_improvement': 'significant' if rows_removed > 0 or (original_missing - remaining_missing) > 0 else 'minimal'
                    }
                    
                    # Extract real cleaning configuration from processing log or session state
                    context['processing_configurations']['cleaning'] = self._extract_cleaning_configuration()
                    
                    if target_step == 'cleaning':
                        context['step_status'] = 'completed'

            # === WEIGHTING STEP CONTEXT ===
            if st.session_state.get('weighted_results') is not None:
                results = st.session_state.weighted_results
                
                # Enhanced weighting results with actual configuration
                context['processing_results']['weighting'] = {
                    'weights_applied': True,
                    'analysis_completed': True,
                    'statistical_results': 'available',
                    'result_type': type(results).__name__
                }
                
                # Extract real weighting configuration
                context['processing_configurations']['weighting'] = self._extract_weighting_configuration(results)
                
                if target_step in ['weighting', 'analysis']:
                    context['step_status'] = 'completed'
            
            # === PROCESSING LOG ANALYSIS ===
            if st.session_state.get('processing_log'):
                recent_logs = st.session_state.processing_log[-5:]  # Last 5 log entries
                context['processing_results']['recent_activity'] = []
                
                for log in recent_logs:
                    if isinstance(log, dict):
                        activity = f"{log.get('action', 'Unknown')}: {log.get('details', 'No details')}"
                        context['processing_results']['recent_activity'].append(activity)
                
                # Step-specific processing history
                context['previous_steps'] = [
                    entry.get('step', 'Unknown') 
                    for entry in st.session_state.processing_log 
                    if isinstance(entry, dict)
                ]
            
            # Build enhanced step summary with real process details
            self._build_enhanced_step_summary(target_step, context)
                
        except Exception as e:
            # Graceful fallback with detailed error info
            context['error'] = f"Context extraction error: {str(e)}"
            context['fallback_mode'] = True
            print(f"Context extraction error for step {target_step}: {str(e)}")
            
        return context
    
    def _extract_cleaning_configuration(self):
        """Extract actual cleaning configuration from session state or processing log."""
        cleaning_config = {
            'methods_used': [],
            'imputation_method': 'None',
            'outlier_detection': [],
            'validation_rules': [],
            'parameters': {}
        }
        
        try:
            # Try to find cleaning configuration in processing log
            if st.session_state.get('processing_log'):
                for log_entry in reversed(st.session_state.processing_log):
                    if isinstance(log_entry, dict) and log_entry.get('step') == 'cleaning':
                        config = log_entry.get('config', {})
                        if config:
                            cleaning_config['imputation_method'] = config.get('missing_method', 'None')
                            cleaning_config['outlier_detection'] = config.get('outlier_methods', [])
                            cleaning_config['parameters'] = {
                                'knn_neighbors': config.get('knn_neighbors', 5),
                                'z_threshold': config.get('z_threshold', 3.0),
                                'winsor_limits': config.get('winsor_limits', 0.1)
                            }
                            if config.get('enable_consistency'):
                                cleaning_config['validation_rules'].append('Consistency checks')
                            if config.get('enable_skip_patterns'):
                                cleaning_config['validation_rules'].append('Skip pattern validation')
                            if config.get('custom_rules'):
                                cleaning_config['validation_rules'].extend(config['custom_rules'])
                            break
            
            # Try to extract from cleaning report if available
            if hasattr(st.session_state, 'cleaning_report') and st.session_state.cleaning_report:
                report = st.session_state.cleaning_report
                cleaning_config['methods_used'] = report.get('steps_performed', [])
                
        except Exception as e:
            print(f"Error extracting cleaning configuration: {str(e)}")
            cleaning_config['extraction_error'] = str(e)
        
        return cleaning_config
    
    def _extract_weighting_configuration(self, results):
        """Extract actual weighting configuration from results and session state."""
        weighting_config = {
            'weight_column': 'None',
            'analysis_variables': [],
            'stratification_variables': [],
            'confidence_level': 0.95,
            'weight_method': 'Equal weights',
            'statistical_methods': []
        }
        
        try:
            # Try to find weighting configuration in processing log
            if st.session_state.get('processing_log'):
                for log_entry in reversed(st.session_state.processing_log):
                    if isinstance(log_entry, dict) and log_entry.get('step') in ['weighting', 'analysis']:
                        config = log_entry.get('config', {})
                        if config:
                            weighting_config['weight_column'] = config.get('weight_column', 'None')
                            weighting_config['analysis_variables'] = config.get('analysis_vars', [])
                            weighting_config['stratification_variables'] = config.get('strat_vars', [])
                            weighting_config['confidence_level'] = config.get('confidence_level', 0.95)
                            break
            
            # Extract statistical methods used from results structure
            if isinstance(results, dict):
                if 'summary_stats' in results:
                    weighting_config['statistical_methods'].append('Descriptive statistics')
                if 'margins_of_error' in results:
                    weighting_config['statistical_methods'].append('Confidence intervals')
                if 'stratified_results' in results:
                    weighting_config['statistical_methods'].append('Stratified analysis')
                if 'weight_diagnostics' in results:
                    weighting_config['statistical_methods'].append('Weight diagnostics')
                    
        except Exception as e:
            print(f"Error extracting weighting configuration: {str(e)}")
            weighting_config['extraction_error'] = str(e)
        
        return weighting_config
    
    def _build_enhanced_step_summary(self, step, context):
        """Build enhanced summary with real process details for cross-step memory."""
        summary = {
            'step': step, 
            'completed': context.get('step_status') == 'completed',
            'process_details': {},
            'configurations': context.get('processing_configurations', {}).get(step, {}),
            'results': context.get('processing_results', {}).get(step, {})
        }
        
        # Step-specific process summaries
        if step == 'upload' and context.get('data_info'):
            data_info = context['data_info']
            summary['process_details'] = {
                'data_loaded': f"{data_info['total_rows']} rows × {data_info['total_columns']} columns",
                'data_quality': f"{data_info['missing_percentage']:.1f}% missing values",
                'data_types': f"{data_info['numeric_columns']} numeric, {data_info['categorical_columns']} categorical"
            }
            
        elif step == 'cleaning' and context.get('processing_results', {}).get('cleaning'):
            cleaning = context['processing_results']['cleaning']
            cleaning_config = context.get('processing_configurations', {}).get('cleaning', {})
            summary['process_details'] = {
                'imputation_method': cleaning_config.get('imputation_method', 'Unknown'),
                'outlier_methods': ', '.join(cleaning_config.get('outlier_detection', [])),
                'rows_processed': f"{cleaning['rows_after_cleaning']} remaining ({cleaning['rows_removed']} removed)",
                'missing_fixed': cleaning['missing_values_fixed'],
                'quality_improvement': cleaning['quality_improvement']
            }
            
        elif step == 'weighting' and context.get('processing_results', {}).get('weighting'):
            weighting_config = context.get('processing_configurations', {}).get('weighting', {})
            summary['process_details'] = {
                'weight_column': weighting_config.get('weight_column', 'None'),
                'analysis_vars': ', '.join(weighting_config.get('analysis_variables', [])),
                'statistical_methods': ', '.join(weighting_config.get('statistical_methods', [])),
                'confidence_level': f"{weighting_config.get('confidence_level', 0.95)*100}%"
            }
        
        # Store in cross-step memory
        if not hasattr(st.session_state, 'cross_step_memory'):
            st.session_state.cross_step_memory = {}
        st.session_state.cross_step_memory[step] = summary
        
        return summary
    
    def _build_step_summary(self, step, context):
        """Build compact summary for cross-step memory."""
        summary = {'step': step, 'completed': context.get('step_status') == 'completed'}
        
        if step == 'upload' and context.get('data_info'):
            summary['summary'] = f"Uploaded {context['data_info']['total_rows']} rows, {context['data_info']['total_columns']} columns"
        elif step == 'cleaning' and context.get('cleaning_results'):
            summary['summary'] = f"Cleaned data: {context['cleaning_results']['rows_removed']} rows removed"
        elif step == 'weighting' and context.get('weighting_info'):
            summary['summary'] = "Applied survey weights and calculated statistics"
        else:
            summary['summary'] = f"Step {step} in progress"
        
        st.session_state.cross_step_memory[step] = summary

    def _create_system_prompt(self, language, current_step):
        """Create step-aware system prompt for the AI based on language and current step."""
        step_context = {
            'upload': 'data upload and initial review',
            'schema': 'data schema configuration',
            'cleaning': 'data cleaning and validation',
            'weighting': 'survey weight application',
            'analysis': 'data analysis and visualization'
        }
        
        current_context = step_context.get(current_step, 'data processing')
        
        prompts = {
            'english': f"""You are a Step-Aware Process Intelligence Assistant for a Survey Data Processing Platform. 

CURRENT STEP: {current_step.upper()} - {current_context}

Your role is to:
1. PRIORITIZE the current step ({current_step}) in your responses
2. Access and reference previous step conversations from cross-step memory when relevant
3. Provide step-specific guidance while maintaining awareness of the full process
4. Answer questions about ANY previous step using stored conversation context
5. Give actionable, step-appropriate suggestions for the current step
6. Bridge information between steps when users ask comparative questions

IMPORTANT: You have access to previous step conversations through cross-step context. When users ask about previous steps, reference this information specifically. Always acknowledge the current step but can answer about any completed step using your memory.

Be conversational, specific with numbers, and seamlessly reference cross-step information when asked.""",

            'hindi': f"""आप एक Step-Aware Survey Data Processing Platform के लिए Process Intelligence Assistant हैं।

वर्तमान चरण: {current_step.upper()} - {current_context}

आपकी भूमिका है:
1. वर्तमान चरण ({current_step}) को प्राथमिकता दें
2. पिछले completed steps को याद रखें और reference करें
3. Step-specific guidance और insights प्रदान करें
4. सभी steps के methodology और results के questions का answer दें
5. Actionable, step-appropriate suggestions दें

हमेशा current processing step को acknowledge करें और contextual help प्रदान करें।""",

            'gujarati': f"""તમે Step-Aware Survey Data Processing Platform માટે Process Intelligence Assistant છો.

હાલનું પગલું: {current_step.upper()} - {current_context}

તમારી ભૂમિકા છે:
1. હાલના પગલા ({current_step}) ને પ્રાથમિકતા આપો
2. પહેલાના completed steps ને યાદ રાખો અને reference કરો
3. Step-specific guidance અને insights આપો
4. બધા steps ના methodology અને results વિશે questions નો answer આપો
5. Actionable, step-appropriate suggestions આપો

હંમેશા current processing step ને acknowledge કરો અને contextual help આપો."""
        }
        
        return prompts.get(language, prompts['english'])
    
    def _get_fallback_response(self, user_message, language, current_step):
        """Provide step-aware fallback responses when AI is unavailable."""
        context = self._get_step_context(current_step)
        
        fallback_responses = {
            'english': {
                'upload': f"I'm here to help with data upload. Current status: {context.get('step_status', 'pending')}. What would you like to know about uploading your survey data?",
                'schema': f"I can help with schema configuration. Your data has {context['data_info'].get('total_columns', 0)} columns. Need help mapping your data structure?",
                'cleaning': f"I'm assisting with data cleaning. Your dataset has {context['data_info'].get('missing_percentage', 0):.1f}% missing values. What cleaning questions do you have?",
                'weighting': f"I can help with survey weighting. Your cleaned data has {context.get('cleaning_results', {}).get('rows_after_cleaning', 0)} rows. Ready for weight application?",
                'analysis': f"I'm here for analysis support. All processing steps are complete. What would you like to analyze or understand about your results?"
            },
            'hindi': {
                'upload': f"मैं data upload में help के लिए हूं। Current status: {context.get('step_status', 'pending')}। Survey data upload के बारे में क्या जानना है?",
                'schema': f"मैं schema configuration में help कर सकता हूं। आपके data में {context['data_info'].get('total_columns', 0)} columns हैं। Data structure mapping में help चाहिए?",
                'cleaning': f"मैं data cleaning में help कर रहा हूं। आपके dataset में {context['data_info'].get('missing_percentage', 0):.1f}% missing values हैं। Cleaning के बारे में क्या questions हैं?",
                'weighting': f"मैं survey weighting में help कर सकता हूं। आपके cleaned data में {context.get('cleaning_results', {}).get('rows_after_cleaning', 0)} rows हैं। Weight application के लिए ready हैं?",
                'analysis': f"मैं analysis support के लिए हूं। सभी processing steps complete हैं। Results के बारे में क्या analyze करना चाहते हैं?"
            },
            'gujarati': {
                'upload': f"હું data upload માં help માટે છું। Current status: {context.get('step_status', 'pending')}. Survey data upload વિશે શું જાણવું છે?",
                'schema': f"હું schema configuration માં help કરી શકું છું। તમારા data માં {context['data_info'].get('total_columns', 0)} columns છે। Data structure mapping માં help જોઈએ?",
                'cleaning': f"હું data cleaning માં help કરી રહ્યો છું। તમારા dataset માં {context['data_info'].get('missing_percentage', 0):.1f}% missing values છે। Cleaning વિશે શું questions છે?",
                'weighting': f"હું survey weighting માં help કરી શકું છું। તમારા cleaned data માં {context.get('cleaning_results', {}).get('rows_after_cleaning', 0)} rows છે। Weight application માટે ready છો?",
                'analysis': f"હું analysis support માટે છું। બધા processing steps complete છે। Results વિશે શું analyze કરવું છે?"
            }
        }
        
        responses = fallback_responses.get(language, fallback_responses['english'])
        return responses.get(current_step, responses['upload'])
    
    def _get_ai_response(self, user_message, language, current_step):
        """Get intelligent response using real data and user intent detection."""
        # First try intelligent rule-based responses for specific data queries
        intelligent_response = self._get_intelligent_response(user_message, current_step)
        if intelligent_response:
            return intelligent_response
            
        # Fall back to AI if available
        if not self.client:
            return self._get_contextual_fallback(user_message, language, current_step)
        
        try:
            context = self._get_step_context(current_step)
            system_prompt = self._create_enhanced_system_prompt(language, current_step, context)
            
            # Get conversation history for context
            chat_history = self._get_recent_chat_history(current_step)
            
            # Create detailed context with real data
            context_str = self._build_comprehensive_context(context, user_message)
            
            # Prepare messages with chat history
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context_str}\n\nChat History: {chat_history}\n\nUser Question: {user_message}"}
            ]
            
            # Call Groq API with better error handling
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=500,
                temperature=0.4,
                timeout=15  # Increased timeout
            )
            
            if response and response.choices and len(response.choices) > 0:
                ai_response = response.choices[0].message.content.strip()
                print(f"✅ GROQ AI response received: {ai_response[:100]}...")
                return ai_response
            else:
                print("❌ Empty response from GROQ API")
                return self._get_contextual_fallback(user_message, language, current_step)
            
        except Exception as e:
            print(f"❌ GROQ API call failed: {str(e)}")
            return self._get_contextual_fallback(user_message, language, current_step)

    def display_chatbot(self):
        """Display the step-aware Process Intelligence Chatbot interface using Streamlit's native chat components."""
        # Simple CSS for chat container styling
        st.markdown("""
        <style>
        .chat-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Update current step based on current page
        self.current_step = self._detect_current_step_from_page(self.current_page)
        
        # Ensure current step has message history
        if self.current_step not in st.session_state.step_chat_messages:
            st.session_state.step_chat_messages[self.current_step] = []
        
        # Header with step awareness
        step_names = {
            'upload': '📁 Data Upload',
            'schema': '🗂️ Schema Setup', 
            'cleaning': '🧹 Data Cleaning',
            'weighting': '⚖️ Weight Application',
            'analysis': '📈 Analysis & Visualization'
        }
        
        current_step_name = step_names.get(self.current_step, self.current_step.title())
        
        # ChatGPT-style header
        st.markdown(f"""
        <div class="chat-header">
            <h3 style="margin: 0; display: flex; align-items: center;">
                🤖 AI Assistant - {current_step_name}
                <span style="margin-left: auto; font-size: 0.8em; opacity: 0.8;">✨ Powered by GROQ</span>
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Step-aware welcome message for new steps (proper transition handling)
        if not st.session_state.step_first_query.get(self.current_step, False):
            welcome_messages = {
                'english': {
                    'upload': "👋 Welcome! I'm your AI assistant for data upload and schema configuration. I'm here to help you upload and review your survey data. What would you like to know?",
                    'cleaning': "🧹 Hello! I'm your AI assistant for data cleaning. I can help you clean, validate, and prepare your data. What cleaning questions do you have?", 
                    'weighting': "⚖️ Hi there! I'm your AI assistant for survey weighting. I can help you apply weights and perform statistical calculations. Ready to discuss weighting strategies?",
                    'analysis': "📈 Welcome! I'm your AI assistant for analysis and visualization. I can help you understand and analyze your results. What would you like to explore?"
                }
            }
            
            language = st.session_state.chatbot_language
            welcome_msg = welcome_messages.get(language, welcome_messages['english']).get(self.current_step, "Hello! How can I help you with your survey data processing?")
            
            # Add contextual information based on previous steps
            if st.session_state.cross_step_memory:
                completed_steps = list(st.session_state.cross_step_memory.keys())
                if completed_steps:
                    welcome_msg += f"\n\n💡 I can also answer questions about previous steps: {', '.join(completed_steps)}"
            
            # Add welcome message to chat
            st.session_state.step_chat_messages[self.current_step].append({
                "role": "assistant", 
                "content": welcome_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.session_state.step_first_query[self.current_step] = True

        # Display chat messages using Streamlit's native chat components
        chat_container = st.container()
        with chat_container:
            # Display all messages for current step only (proper isolation)
            for message in st.session_state.step_chat_messages[self.current_step]:
                with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
                    st.write(message["content"])
                    st.caption(f"_{message['timestamp']}_")

        # ChatGPT-style input with enhanced placeholder
        placeholder_messages = {
            'upload': 'Ask me about uploading your survey data...',
            'schema': 'Ask me about configuring your data schema...',
            'cleaning': 'Ask me about cleaning and validating your data...',
            'weighting': 'Ask me about applying survey weights...',
            'analysis': 'Ask me about analyzing your results...'
        }
        
        placeholder = placeholder_messages.get(self.current_step, 'Type your question here...')
        
        # Chat input with immediate response rendering
        if prompt := st.chat_input(placeholder):
            # Add user message to chat and display immediately
            user_msg = {
                "role": "user", 
                "content": prompt,
                "timestamp": datetime.now().strftime("%H:%M")
            }
            st.session_state.step_chat_messages[self.current_step].append(user_msg)
            
            # Display user message immediately
            with st.chat_message("user", avatar="👤"):
                st.write(prompt)
                st.caption(f"_{user_msg['timestamp']}_")
            
            # Generate and display AI response immediately
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("AI Assistant is thinking..."):
                    language = st.session_state.chatbot_language
                    ai_response = self._get_ai_response(prompt, language, self.current_step)
                
                st.write(ai_response)
                response_time = datetime.now().strftime("%H:%M")
                st.caption(f"_{response_time}_")
                
                # Add AI response to chat history
                assistant_msg = {
                    "role": "assistant", 
                    "content": ai_response,
                    "timestamp": response_time
                }
                st.session_state.step_chat_messages[self.current_step].append(assistant_msg)
            
            # Force a rerun to show the new messages properly
            st.rerun()

        # Enhanced sidebar with ChatGPT-style options
        with st.sidebar:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="margin: 0; text-align: center;">🤖 AI Chat Settings</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Language selector with improved styling
            st.markdown("**🌐 Chat Language**")
            language_options = {
                '🇺🇸 English': 'english',
                '🇮🇳 हिन्दी': 'hindi', 
                '🇮🇳 ગુજરાતી': 'gujarati'
            }
            
            selected_lang = st.selectbox(
                "Select Language",
                options=list(language_options.keys()),
                index=list(language_options.values()).index(st.session_state.chatbot_language),
                key="chatbot_language_selector"
            )
            st.session_state.chatbot_language = language_options[selected_lang]
            
            # Quick action buttons
            st.markdown("---")
            st.markdown("**⚡ Quick Actions**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📈 Data Summary", key="quick_summary"):
                    summary_prompt = "Can you give me a quick summary of my current data and processing status?"
                    self._add_quick_message(summary_prompt)
                    
            with col2:
                if st.button("📝 Tips", key="quick_tips"):
                    tips_prompt = "What are some best practices and tips for this current step?"
                    self._add_quick_message(tips_prompt)

        # Cross-step navigation in sidebar with improved styling
        with st.sidebar:
            st.markdown("---")
            st.markdown("**📋 Processing Journey**")
            
            # Progress visualization
            completed_steps = sum(1 for summary in st.session_state.cross_step_memory.values() if summary.get('completed', False))
            total_steps = len(st.session_state.cross_step_memory) if st.session_state.cross_step_memory else 5
            
            if total_steps > 0:
                progress = completed_steps / total_steps
                st.progress(progress)
                st.write(f"✨ Progress: {completed_steps}/{total_steps} steps completed")
            
            # Step history with enhanced display
            for step, summary in st.session_state.cross_step_memory.items():
                status_icon = "✅" if summary.get('completed', False) else "⏳"
                current_icon = "➡️" if step == self.current_step else ""
                st.markdown(f"{status_icon}{current_icon} **{step.title()}**")
                st.caption(summary.get('summary', 'In progress...'))

        # Chat management options
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🗑️ Clear Chat", key="clear_step_chat", help="Clear current step's chat history"):
                st.session_state.step_chat_messages[self.current_step] = []
                if self.current_step in st.session_state.step_first_query:
                    del st.session_state.step_first_query[self.current_step]
                st.success("Chat cleared! ✨")
        
        with col2:
            # Export chat button
            if st.button("💾 Export Chat", key="export_chat", help="Export chat history"):
                chat_history = st.session_state.step_chat_messages.get(self.current_step, [])
                if chat_history:
                    chat_text = "\n\n".join([f"{msg['timestamp']} - {msg['role'].title()}: {msg['content']}" for msg in chat_history])
                    st.download_button(
                        label="Download Chat History",
                        data=chat_text,
                        file_name=f"chat_history_{self.current_step}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("No chat history to export")
    
    def _add_quick_message(self, message):
        """Add a quick action message to the chat."""
        user_msg = {
            "role": "user", 
            "content": message,
            "timestamp": datetime.now().strftime("%H:%M")
        }
        st.session_state.step_chat_messages[self.current_step].append(user_msg)
        
        # Generate AI response
        language = st.session_state.chatbot_language
        ai_response = self._get_ai_response(message, language, self.current_step)
        
        # Add AI response to chat
        assistant_msg = {
            "role": "assistant", 
            "content": ai_response,
            "timestamp": datetime.now().strftime("%H:%M")
        }
        st.session_state.step_chat_messages[self.current_step].append(assistant_msg)
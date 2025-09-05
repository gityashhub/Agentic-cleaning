import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from groq import Groq
import traceback

class ProcessIntelligenceChatbot:
    """Multilingual AI Assistant that remembers the entire data processing workflow."""
    
    def __init__(self):
        # Initialize Groq client
        self.client = None
        self._init_groq_client()
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Language settings
        if 'chatbot_language' not in st.session_state:
            st.session_state.chatbot_language = 'english'
            
        # Context cache
        self._context_cache = None
        
    def _init_groq_client(self):
        """Initialize Groq client with error handling."""
        try:
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                self.client = Groq(api_key=api_key)
            else:
                st.error("⚠️ Groq API key not found. Chatbot will use fallback responses.")
        except Exception as e:
            st.error(f"⚠️ Could not initialize AI client: {str(e)}")
            self.client = None
    
    def _get_processing_context(self):
        """Extract comprehensive processing context from session state."""
        if self._context_cache:
            return self._context_cache
            
        context = {
            'data_info': {},
            'processing_steps': [],
            'quality_metrics': {},
            'current_state': 'initial',
            'audit_trail': [],
            'recommendations': []
        }
        
        try:
            # Data information
            if st.session_state.get('data') is not None:
                data = st.session_state.data
                context['data_info'] = {
                    'total_rows': int(data.shape[0]),
                    'total_columns': int(data.shape[1]),
                    'missing_values': int(data.isnull().sum().sum()),
                    'missing_percentage': float((data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100),
                    'numeric_columns': int(len(data.select_dtypes(include=[np.number]).columns)),
                    'categorical_columns': int(len(data.select_dtypes(include=['object']).columns)),
                    'duplicates': int(data.duplicated().sum()) if hasattr(data, 'duplicated') else 0
                }
                context['current_state'] = 'data_uploaded'
            
            # Cleaned data information
            if st.session_state.get('cleaned_data') is not None:
                cleaned = st.session_state.cleaned_data
                original_rows = context['data_info'].get('total_rows', 0)
                rows_removed = original_rows - cleaned.shape[0] if original_rows > 0 else 0
                
                context['cleaning_results'] = {
                    'rows_after_cleaning': int(cleaned.shape[0]),
                    'rows_removed': int(rows_removed),
                    'removal_percentage': float((rows_removed / original_rows * 100)) if original_rows > 0 else 0,
                    'remaining_missing': int(cleaned.isnull().sum().sum()),
                    'quality_improvement': 'significant' if rows_removed > 0 else 'minimal'
                }
                context['current_state'] = 'data_cleaned'
            
            # Weighted results
            if st.session_state.get('weighted_results') is not None:
                results = st.session_state.weighted_results
                context['weighting_info'] = {
                    'weights_applied': True,
                    'analysis_completed': True,
                    'statistical_results': 'available'
                }
                context['current_state'] = 'analysis_complete'
            
            # Processing log
            if st.session_state.get('processing_log'):
                context['processing_steps'] = [
                    {
                        'step': entry.get('step', 'Unknown'),
                        'timestamp': entry.get('timestamp', datetime.now()).strftime('%H:%M:%S'),
                        'config': entry.get('config', {}),
                        'success': True
                    }
                    for entry in st.session_state.processing_log
                ]
            
            # Audit trail
            if st.session_state.get('audit_log'):
                context['audit_trail'] = [
                    {
                        'action': entry.get('action_type', 'Unknown'),
                        'timestamp': entry.get('timestamp', ''),
                        'details': entry.get('details', '')
                    }
                    for entry in st.session_state.audit_log[-10:]  # Last 10 entries
                ]
                
        except Exception as e:
            # Graceful fallback
            context['error'] = f"Context extraction error: {str(e)}"
            
        self._context_cache = context
        return context
    
    def _create_system_prompt(self, language):
        """Create system prompt for the AI based on language."""
        prompts = {
            'english': """You are a Process Intelligence Assistant for a Survey Data Processing Platform. You have complete memory of all data processing steps performed by the user. 

Your role is to:
1. Explain what was done during data processing in clear, conversational language
2. Assess data quality and processing effectiveness  
3. Suggest improvements and optimizations
4. Answer questions about methodology and results
5. Provide actionable insights

Always respond in a helpful, professional tone. Be specific about numbers, percentages, and technical details when relevant. If asked about improvements, provide concrete suggestions.""",

            'hindi': """आप एक Survey Data Processing Platform के लिए Process Intelligence Assistant हैं। आपके पास user द्वारा किए गए सभी data processing steps की complete memory है।

आपकी भूमिका है:
1. Data processing के दौरान क्या किया गया, उसे clear और conversational language में explain करना
2. Data quality और processing effectiveness का assessment करना
3. Improvements और optimizations suggest करना  
4. Methodology और results के बारे में questions का answer देना
5. Actionable insights provide करना

हमेशा helpful और professional tone में respond करें। Numbers, percentages और technical details के साथ specific रहें। अगर improvements के बारे में पूछा जाए तो concrete suggestions दें।""",

            'gujarati': """તમે Survey Data Processing Platform માટે Process Intelligence Assistant છો. તમારી પાસે user દ્વારા કરવામાં આવેલા તમામ data processing steps ની complete memory છે.

તમારી ભૂમિકા છે:
1. Data processing દરમિયાન શું કરવામાં આવ્યું તે clear અને conversational language માં explain કરવું
2. Data quality અને processing effectiveness નું assessment કરવું
3. Improvements અને optimizations suggest કરવા
4. Methodology અને results વિશે questions નો answer આપવો
5. Actionable insights provide કરવા

હંમેશા helpful અને professional tone માં respond કરો. Numbers, percentages અને technical details સાથે specific રહો. જો improvements વિશે પૂછવામાં આવે તો concrete suggestions આપો."""
        }
        
        return prompts.get(language, prompts['english'])
    
    def _get_fallback_response(self, user_message, language):
        """Provide fallback responses when AI is unavailable."""
        context = self._get_processing_context()
        
        fallback_responses = {
            'english': {
                'summary': f"Your dataset has {context['data_info'].get('total_rows', 0)} rows and {context['data_info'].get('total_columns', 0)} columns. Current processing state: {context['current_state'].replace('_', ' ').title()}.",
                'quality': f"Data quality metrics: {context['data_info'].get('missing_percentage', 0):.1f}% missing values, {context['data_info'].get('duplicates', 0)} duplicates found.",
                'help': "I can help you understand your data processing workflow. Try asking specific questions about cleaning, weighting, or analysis results.",
                'default': "I'm here to help with your survey data processing. What would you like to know about your data or the processing steps?"
            },
            'hindi': {
                'summary': f"आपके dataset में {context['data_info'].get('total_rows', 0)} rows और {context['data_info'].get('total_columns', 0)} columns हैं। Current processing state: {context['current_state'].replace('_', ' ').title()}।",
                'quality': f"Data quality metrics: {context['data_info'].get('missing_percentage', 0):.1f}% missing values, {context['data_info'].get('duplicates', 0)} duplicates मिले।",
                'help': "मैं आपके data processing workflow को समझने में help कर सकता हूं। Cleaning, weighting, या analysis results के बारे में specific questions पूछें।",
                'default': "मैं आपके survey data processing में help के लिए यहां हूं। आपको अपने data या processing steps के बारे में क्या जानना है?"
            },
            'gujarati': {
                'summary': f"તમારા dataset માં {context['data_info'].get('total_rows', 0)} rows અને {context['data_info'].get('total_columns', 0)} columns છે. Current processing state: {context['current_state'].replace('_', ' ').title()}.",
                'quality': f"Data quality metrics: {context['data_info'].get('missing_percentage', 0):.1f}% missing values, {context['data_info'].get('duplicates', 0)} duplicates મળ્યા.",
                'help': "હું તમારા data processing workflow ને સમજવામાં help કરી શકું છું. Cleaning, weighting, અથવા analysis results વિશે specific questions પૂછો.",
                'default': "હું તમારા survey data processing માં help માટે અહીં છું. તમને તમારા data અથવા processing steps વિશે શું જાણવું છે?"
            }
        }
        
        responses = fallback_responses.get(language, fallback_responses['english'])
        
        # Determine response type based on message content
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ['what', 'શું', 'क्या', 'summary', 'done', 'किया', 'કર્યું']):
            return responses['summary']
        elif any(word in message_lower for word in ['quality', 'problem', 'issue', 'समस्या', 'પ્રોબ્લેમ']):
            return responses['quality']
        elif any(word in message_lower for word in ['help', 'how', 'कैसे', 'કેવી']):
            return responses['help']
        else:
            return responses['default']
    
    def _get_ai_response(self, user_message, language):
        """Get response from Groq AI with comprehensive error handling."""
        if not self.client:
            return self._get_fallback_response(user_message, language)
        
        try:
            context = self._get_processing_context()
            system_prompt = self._create_system_prompt(language)
            
            # Create context string
            context_str = f"""
            Current Processing Context:
            - Data Info: {json.dumps(context.get('data_info', {}), indent=2)}
            - Processing State: {context.get('current_state', 'unknown')}
            - Processing Steps: {json.dumps(context.get('processing_steps', []), indent=2)}
            - Cleaning Results: {json.dumps(context.get('cleaning_results', {}), indent=2)}
            - Weighting Info: {json.dumps(context.get('weighting_info', {}), indent=2)}
            """
            
            # Prepare messages for Groq
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context_str}\n\nUser Question: {user_message}"}
            ]
            
            # Call Groq API with optimized settings
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Fastest model for better performance
                messages=messages,
                max_tokens=800,  # Reduced for faster responses
                temperature=0.6,
                timeout=8  # 8 second timeout for faster response
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Log error and provide fallback
            error_msg = f"AI service temporarily unavailable: {str(e)}"
            st.warning(f"⚠️ {error_msg}")
            return self._get_fallback_response(user_message, language)
    
    def display_chatbot(self):
        """Display the Process Intelligence Chatbot interface."""
        st.markdown("---")
        st.subheader("🤖 Process Intelligence Assistant")
        st.markdown("*Ask me anything about your data processing workflow in English, Hindi, or Gujarati*")
        
        # Language selector and status
        col1, col2 = st.columns([3, 1])
        
        with col2:
            language_options = {
                'English': 'english',
                'हिन्दी': 'hindi', 
                'ગુજરાતી': 'gujarati'
            }
            
            selected_lang = st.selectbox(
                "Language / भाषा / ભાષા",
                options=list(language_options.keys()),
                index=list(language_options.values()).index(st.session_state.chatbot_language)
            )
            st.session_state.chatbot_language = language_options[selected_lang]
        
        with col1:
            # Processing status summary
            context = self._get_processing_context()
            status_text = self._get_status_summary(context, st.session_state.chatbot_language)
            st.markdown(f"**📊 Current Status:** {status_text}")
        
        # Chat interface
        st.markdown("**💬 Chat with Your Processing Assistant:**")
        
        # Display chat history (last 6 messages for performance)
        if st.session_state.chat_history:
            with st.container():
                chat_display = st.session_state.chat_history[-6:]  # Limit for performance
                for i, (role, message, timestamp, lang) in enumerate(chat_display):
                    if role == "user":
                        st.markdown(f"**👤 You ({timestamp}):** {message}")
                    else:
                        st.markdown(f"**🤖 Assistant ({timestamp}):** {message}")
                    
                    # Add separator for readability
                    if i < len(chat_display) - 1:
                        st.markdown("---")
        
        # Chat input with language-appropriate placeholder
        placeholders = {
            'english': "Ask me: What did you do to my data?",
            'hindi': "मुझसे पूछें: मेरे डेटा के साथ क्या किया गया?",
            'gujarati': "મને પૂછો: મારા ડેટા સાથે શું કર્યું?"
        }
        
        user_input = st.text_input(
            "Ask about your data processing:",
            placeholder=placeholders.get(st.session_state.chatbot_language, placeholders['english']),
            key="chatbot_input"
        )
        
        # Process user input
        if user_input:
            self._process_user_message(user_input)
            st.rerun()
        
        # Quick action buttons
        self._display_quick_actions()
        
        # Clear chat option
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            self._context_cache = None  # Reset context cache
            st.rerun()
    
    def _process_user_message(self, user_message):
        """Process user message and generate AI response."""
        timestamp = datetime.now().strftime("%H:%M")
        language = st.session_state.chatbot_language
        
        # Add user message to history
        st.session_state.chat_history.append(("user", user_message, timestamp, language))
        
        # Generate AI response
        with st.spinner("🤔 Thinking..."):
            ai_response = self._get_ai_response(user_message, language)
        
        # Add AI response to history
        st.session_state.chat_history.append(("assistant", ai_response, timestamp, language))
        
        # Clear context cache to get fresh data on next request
        self._context_cache = None
    
    def _get_status_summary(self, context, language):
        """Get status summary in specified language."""
        summaries = {
            'english': {
                'initial': "No data uploaded yet",
                'data_uploaded': f"{context['data_info'].get('total_rows', 0)} rows uploaded",
                'data_cleaned': f"Data cleaned, {context.get('cleaning_results', {}).get('rows_after_cleaning', 0)} rows remaining",
                'analysis_complete': "Full analysis completed with weights applied"
            },
            'hindi': {
                'initial': "अभी तक कोई डेटा upload नहीं हुआ",
                'data_uploaded': f"{context['data_info'].get('total_rows', 0)} rows upload हुए",
                'data_cleaned': f"डेटा clean हुआ, {context.get('cleaning_results', {}).get('rows_after_cleaning', 0)} rows बचे",
                'analysis_complete': "Weights के साथ पूरा analysis complete हुआ"
            },
            'gujarati': {
                'initial': "હજુ સુધી કોઈ ડેટા upload થયો નથી",
                'data_uploaded': f"{context['data_info'].get('total_rows', 0)} rows upload થયા",
                'data_cleaned': f"ડેટા clean થયો, {context.get('cleaning_results', {}).get('rows_after_cleaning', 0)} rows બાકી",
                'analysis_complete': "Weights સાથે પૂરો analysis complete થયો"
            }
        }
        
        state = context.get('current_state', 'initial')
        return summaries.get(language, summaries['english']).get(state, state)
    
    def _display_quick_actions(self):
        """Display quick action buttons for common queries."""
        st.markdown("**🚀 Quick Questions:**")
        
        # Language-specific quick actions
        if st.session_state.chatbot_language == 'english':
            actions = [
                ("📊 What was done?", "What data processing steps were performed on my dataset?"),
                ("❓ Any problems?", "Are there any data quality issues or problems I should know about?"),
                ("💡 Improvements?", "What improvements or optimizations do you recommend?"),
                ("📈 How accurate?", "How accurate and reliable are my analysis results?")
            ]
        elif st.session_state.chatbot_language == 'hindi':
            actions = [
                ("📊 क्या किया गया?", "मेरे dataset पर कौन से data processing steps किए गए?"),
                ("❓ कोई समस्या?", "क्या कोई data quality issues या problems हैं जिनके बारे में मुझे पता होना चाहिए?"),
                ("💡 सुधार?", "आप कौन से improvements या optimizations recommend करते हैं?"),
                ("📈 कितना सटीक?", "मेरे analysis results कितने accurate और reliable हैं?")
            ]
        else:  # gujarati
            actions = [
                ("📊 શું કર્યું?", "મારા dataset પર કયા data processing steps કરવામાં આવ્યા?"),
                ("❓ કોઈ સમસ્યા?", "શું કોઈ data quality issues અથવા problems છે જેની મને જાણકારી હોવી જોઈએ?"),
                ("💡 સુધારા?", "તમે કયા improvements અથવા optimizations recommend કરો છો?"),
                ("📈 કેટલું accurate?", "મારા analysis results કેટલા accurate અને reliable છે?")
            ]
        
        # Display buttons in a grid
        cols = st.columns(2)
        for i, (button_text, question) in enumerate(actions):
            with cols[i % 2]:
                if st.button(button_text, key=f"quick_action_{i}"):
                    self._process_user_message(question)
                    st.rerun()
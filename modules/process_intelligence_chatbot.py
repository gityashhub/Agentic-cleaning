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
    
    def __init__(self, current_step=None):
        # Current processing step awareness
        self.current_step = current_step or self._detect_current_step()
        
        # Initialize Groq client
        self.client = None
        self._init_groq_client()
        
        # Initialize step-aware chat history
        if 'step_chat_messages' not in st.session_state:
            st.session_state.step_chat_messages = {}
        
        # Ensure current step has message history
        if self.current_step not in st.session_state.step_chat_messages:
            st.session_state.step_chat_messages[self.current_step] = []
        
        # Initialize cross-step memory
        if 'cross_step_memory' not in st.session_state:
            st.session_state.cross_step_memory = {}
        
        # Language settings
        if 'chatbot_language' not in st.session_state:
            st.session_state.chatbot_language = 'english'
        
        # Track first query per step
        if 'step_first_query' not in st.session_state:
            st.session_state.step_first_query = {}
            
        # Context cache with step awareness
        self._context_cache = {}
        self._step_summaries = {}

    def _detect_current_step(self):
        """Detect current processing step based on session state."""
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
    
    def _init_groq_client(self):
        """Initialize Groq client with silent error handling."""
        try:
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                self.client = Groq(api_key=api_key)
            else:
                self.client = None  # Will use fallback responses
        except Exception:
            self.client = None  # Silent fallback

    def _get_step_context(self, step=None):
        """Get context specific to a processing step."""
        target_step = step or self.current_step
        
        if target_step in self._context_cache:
            return self._context_cache[target_step]
            
        context = {
            'step': target_step,
            'data_info': {},
            'step_status': 'pending',
            'previous_steps': [],
            'next_steps': [],
            'recommendations': []
        }
        
        try:
            # Data information with safe handling
            if st.session_state.get('data') is not None:
                data = st.session_state.data
                if data is not None and hasattr(data, 'shape'):
                    context['data_info'] = {
                        'total_rows': int(data.shape[0]),
                        'total_columns': int(data.shape[1]),
                        'missing_values': int(data.isnull().sum().sum()),
                        'missing_percentage': float((data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100),
                        'numeric_columns': int(len(data.select_dtypes(include=[np.number]).columns)),
                        'categorical_columns': int(len(data.select_dtypes(include=['object']).columns)),
                        'duplicates': int(data.duplicated().sum()) if hasattr(data, 'duplicated') else 0
                    }
                    context['step_status'] = 'completed' if target_step == 'upload' else 'available'
            
            # Cleaned data information with safe handling
            if st.session_state.get('cleaned_data') is not None:
                cleaned = st.session_state.cleaned_data
                if cleaned is not None and hasattr(cleaned, 'shape'):
                    original_rows = context['data_info'].get('total_rows', 0)
                    rows_removed = original_rows - cleaned.shape[0] if original_rows > 0 else 0
                    
                    context['cleaning_results'] = {
                        'rows_after_cleaning': int(cleaned.shape[0]),
                        'rows_removed': int(rows_removed),
                        'removal_percentage': float((rows_removed / original_rows * 100)) if original_rows > 0 else 0,
                        'remaining_missing': int(cleaned.isnull().sum().sum()),
                        'quality_improvement': 'significant' if rows_removed > 0 else 'minimal'
                    }
                    if target_step == 'cleaning':
                        context['step_status'] = 'completed'

            # Weighted results
            if st.session_state.get('weighted_results') is not None:
                results = st.session_state.weighted_results
                context['weighting_info'] = {
                    'weights_applied': True,
                    'analysis_completed': True,
                    'statistical_results': 'available'
                }
                if target_step in ['weighting', 'analysis']:
                    context['step_status'] = 'completed'
            
            # Processing log with step awareness
            if st.session_state.get('processing_log'):
                context['previous_steps'] = [
                    entry.get('step', 'Unknown')
                    for entry in st.session_state.processing_log
                ]
            
            # Build step summary for cross-step memory
            self._build_step_summary(target_step, context)
                
        except Exception as e:
            # Graceful fallback
            context['error'] = f"Context extraction error: {str(e)}"
            
        self._context_cache[target_step] = context
        return context
    
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
2. Remember and reference previous completed steps when relevant
3. Provide step-specific guidance and insights
4. Answer questions about methodology and results across all steps
5. Give actionable, step-appropriate suggestions

Always acknowledge the current processing step and provide contextual help. Be conversational, specific with numbers, and reference cross-step information when asked.""",

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
        """Get response from Groq AI with comprehensive error handling."""
        if not self.client:
            return self._get_fallback_response(user_message, language, current_step)
        
        try:
            context = self._get_step_context(current_step)
            system_prompt = self._create_system_prompt(language, current_step)
            
            # Create compact context string to avoid token limits
            context_str = f"""
            Step: {current_step}
            Status: {context.get('step_status', 'pending')}
            Data: {context.get('data_info', {})}
            Previous Steps: {list(st.session_state.cross_step_memory.keys())}
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
                max_tokens=400,  # Reduced for faster responses
                temperature=0.6,
                timeout=8  # 8 second timeout for faster response
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Log error and provide fallback
            return self._get_fallback_response(user_message, language, current_step)

    def display_chatbot(self):
        """Display the ChatGPT-style Process Intelligence Chatbot interface."""
        # Add ChatGPT-style CSS
        st.markdown("""
        <style>
        .chat-container {
            background-color: #f7f7f8;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            max-height: 600px;
            overflow-y: auto;
        }
        
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 16px;
            border-radius: 18px;
            margin: 8px 0;
            max-width: 80%;
            margin-left: auto;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .assistant-message {
            background: white;
            color: #333;
            padding: 12px 16px;
            border-radius: 18px;
            margin: 8px 0;
            max-width: 80%;
            border: 1px solid #e1e5e9;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .chat-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 10px 10px 0 0;
            margin-bottom: 0;
        }
        
        .timestamp {
            font-size: 0.8em;
            opacity: 0.7;
            margin-top: 5px;
        }
        
        .typing-indicator {
            display: flex;
            align-items: center;
            padding: 10px 16px;
            background: #f1f3f4;
            border-radius: 18px;
            margin: 8px 0;
            max-width: 80%;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Update current step
        self.current_step = self._detect_current_step()
        
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
        
        # Step-aware welcome message for first query
        if self.current_step not in st.session_state.step_first_query:
            welcome_messages = {
                'english': {
                    'upload': "👋 Welcome! I'm your AI assistant for survey data processing. I'm here to help you upload and review your survey data. What would you like to know?",
                    'schema': "🗂️ Great! Now I can help you configure your data schema. Any questions about your data structure?", 
                    'cleaning': "🧹 Perfect! I'm ready to assist with data cleaning. What cleaning questions do you have?",
                    'weighting': "⚖️ Excellent! I can help you apply survey weights. Ready to discuss weighting strategies?",
                    'analysis': "📈 Fantastic! All steps are complete. I can help you understand and analyze your results."
                }
            }
            
            language = st.session_state.chatbot_language
            welcome_msg = welcome_messages.get(language, welcome_messages['english']).get(self.current_step, "Hello! How can I help you with your survey data processing?")
            
            # Add welcome message to chat
            st.session_state.step_chat_messages[self.current_step].append({
                "role": "assistant", 
                "content": welcome_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.session_state.step_first_query[self.current_step] = True

        # ChatGPT-style chat container
        with st.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Display chat messages with ChatGPT-style formatting
            for message in st.session_state.step_chat_messages[self.current_step]:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                        <div class="user-message">
                            <strong>👤 You</strong><br>
                            {message["content"]}
                            <div class="timestamp">{message['timestamp']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                        <div class="assistant-message">
                            <strong>🤖 AI Assistant</strong><br>
                            {message["content"]}
                            <div class="timestamp">{message['timestamp']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        # ChatGPT-style input with enhanced placeholder
        placeholder_messages = {
            'upload': 'Ask me about uploading your survey data...',
            'schema': 'Ask me about configuring your data schema...',
            'cleaning': 'Ask me about cleaning and validating your data...',
            'weighting': 'Ask me about applying survey weights...',
            'analysis': 'Ask me about analyzing your results...'
        }
        
        placeholder = placeholder_messages.get(self.current_step, 'Type your question here...')
        
        if prompt := st.chat_input(placeholder):
            # Add user message to chat
            user_msg = {
                "role": "user", 
                "content": prompt,
                "timestamp": datetime.now().strftime("%H:%M")
            }
            st.session_state.step_chat_messages[self.current_step].append(user_msg)
            
            # Show typing indicator
            with st.empty():
                st.markdown("""
                <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                    <div class="typing-indicator">
                        <strong>🤖 AI Assistant</strong> is typing
                        <span style="margin-left: 10px;">• • •</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Generate AI response
            language = st.session_state.chatbot_language
            ai_response = self._get_ai_response(prompt, language, self.current_step)
            
            # Add AI response to chat
            assistant_msg = {
                "role": "assistant", 
                "content": ai_response,
                "timestamp": datetime.now().strftime("%H:%M")
            }
            st.session_state.step_chat_messages[self.current_step].append(assistant_msg)

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
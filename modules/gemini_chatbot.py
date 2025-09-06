import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import hashlib
import time
from typing import Dict, List, Optional, Any
import traceback

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class GeminiChatbot:
    """
    Central, intelligent chatbot using Google Gemini API with step-aware memory management,
    context pruning, and efficient token usage for survey data processing platform.
    """
    
    def __init__(self, current_step=None, current_page=None):
        # Current processing step awareness
        self.current_page = current_page
        self.current_step = current_step or self._detect_current_step_from_page(current_page)
        
        # Initialize Gemini client
        self.client = None
        self.model = None
        self._init_gemini_client()
        
        # Token management and efficiency
        self.max_context_tokens = 25000  # Conservative limit for Gemini free tier
        self.response_tokens = 2000      # Reserve tokens for response
        self.available_tokens = self.max_context_tokens - self.response_tokens
        
        # Initialize comprehensive memory system
        self._init_memory_system()
        
        # Context management and pruning
        self._context_cache = {}
        self._token_usage_tracker = {}
        
        # Fallback system
        self._fallback_responses = self._load_fallback_responses()
        
        # Language settings
        if 'chatbot_language' not in st.session_state:
            st.session_state.chatbot_language = 'english'
    
    def _init_gemini_client(self):
        """Initialize Gemini client with proper error handling and API key management."""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key and api_key.strip() and GEMINI_AVAILABLE:
                genai.configure(api_key=api_key.strip())
                
                # Initialize model with optimized settings for accuracy and efficiency
                generation_config = {
                    "temperature": 0.1,  # Lower temperature for more accurate, consistent responses
                    "top_p": 0.8,
                    "top_k": 20,
                    "max_output_tokens": 2000,
                    "response_mime_type": "text/plain",
                }
                
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
                
                self.model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash-8b",  # Using free tier model
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
                
                # Test connection
                test_response = self.model.generate_content("Hello")
                if test_response and test_response.text:
                    print("✅ Gemini client initialized successfully")
                    self.client = True
                else:
                    self.client = None
                    print("❌ Gemini client test failed")
            else:
                if not GEMINI_AVAILABLE:
                    print("❌ Google Generative AI package not available")
                else:
                    print("❌ No GEMINI_API_KEY found")
                self.client = None
        except Exception as e:
            print(f"❌ Gemini client initialization failed: {str(e)}")
            self.client = None
    
    def _init_memory_system(self):
        """Initialize comprehensive step-wise memory and context storage system."""
        # Step-specific chat histories
        if 'step_chat_messages' not in st.session_state:
            st.session_state.step_chat_messages = {}
        
        # Cross-step memory for context continuity
        if 'cross_step_memory' not in st.session_state:
            st.session_state.cross_step_memory = {}
        
        # Step-wise summaries for efficient context management
        if 'step_summaries' not in st.session_state:
            st.session_state.step_summaries = {}
        
        # Processing configurations memory
        if 'processing_memory' not in st.session_state:
            st.session_state.processing_memory = {
                'data_info': {},
                'configurations': {},
                'results': {},
                'decisions': {},
                'insights': {}
            }
        
        # Current active step tracking
        if 'current_active_step' not in st.session_state:
            st.session_state.current_active_step = None
        
        # Context pruning history for optimization
        if 'context_pruning_log' not in st.session_state:
            st.session_state.context_pruning_log = []
        
        # Initialize current step
        self._initialize_step_context()
    
    def _initialize_step_context(self):
        """Initialize context for current step with proper transition handling."""
        # Detect step transition
        if st.session_state.current_active_step != self.current_step:
            # Store previous step's context before switching
            if st.session_state.current_active_step is not None:
                self._create_step_summary(st.session_state.current_active_step)
            
            # Update active step
            st.session_state.current_active_step = self.current_step
        
        # Ensure current step has message history
        if self.current_step not in st.session_state.step_chat_messages:
            st.session_state.step_chat_messages[self.current_step] = []
    
    def _create_step_summary(self, step: str):
        """Create intelligent summary of step for cross-step memory."""
        messages = st.session_state.step_chat_messages.get(step, [])
        if not messages:
            return
        
        # Extract key information from conversations and processing
        summary_data = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'conversation_count': len(messages),
            'key_topics': self._extract_conversation_topics(messages),
            'decisions_made': self._extract_decisions(messages),
            'data_insights': self._extract_data_insights(step),
            'processing_details': self._extract_processing_details(step),
            'user_concerns': self._extract_user_concerns(messages),
            'recommendations_given': self._extract_recommendations(messages)
        }
        
        # Store in step summaries
        st.session_state.step_summaries[step] = summary_data
        
        # Update cross-step memory
        st.session_state.cross_step_memory[step] = {
            'completed': True,
            'summary': summary_data,
            'last_activity': datetime.now().isoformat()
        }
    
    def _extract_conversation_topics(self, messages: List[Dict]) -> List[str]:
        """Extract key topics from conversation using keyword analysis."""
        topics = set()
        keywords_map = {
            'data_upload': ['upload', 'file', 'import', 'csv', 'excel'],
            'data_quality': ['missing', 'null', 'quality', 'clean', 'duplicate'],
            'data_cleaning': ['imputation', 'outlier', 'validation', 'knn', 'median'],
            'survey_weights': ['weight', 'survey', 'sampling', 'representative'],
            'analysis': ['analysis', 'statistical', 'correlation', 'trend'],
            'visualization': ['chart', 'graph', 'plot', 'visual'],
            'reporting': ['report', 'generate', 'export', 'pdf']
        }
        
        for msg in messages:
            if msg['role'] == 'user':
                content = msg['content'].lower()
                for topic, keywords in keywords_map.items():
                    if any(keyword in content for keyword in keywords):
                        topics.add(topic)
        
        return list(topics)
    
    def _extract_decisions(self, messages: List[Dict]) -> List[str]:
        """Extract decisions and choices made during the conversation."""
        decisions = []
        decision_indicators = ['chose', 'selected', 'decided', 'will use', 'prefer']
        
        for msg in messages:
            content = msg['content'].lower()
            if any(indicator in content for indicator in decision_indicators):
                decisions.append(msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content'])
        
        return decisions[-3:]  # Keep last 3 decisions
    
    def _extract_data_insights(self, step: str) -> Dict[str, Any]:
        """Extract data insights specific to the step from session state."""
        insights = {}
        
        if hasattr(st.session_state, 'data') and st.session_state.data is not None:
            data = st.session_state.data
            insights['rows'] = len(data)
            insights['columns'] = len(data.columns)
            insights['missing_percentage'] = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            
            if step == 'cleaning' and hasattr(st.session_state, 'cleaned_data'):
                cleaned = st.session_state.cleaned_data
                if cleaned is not None:
                    insights['rows_after_cleaning'] = len(cleaned)
                    insights['rows_removed'] = len(data) - len(cleaned)
            
            if step == 'weighting' and hasattr(st.session_state, 'weighted_results'):
                insights['weights_applied'] = st.session_state.weighted_results is not None
        
        return insights
    
    def _extract_processing_details(self, step: str) -> Dict[str, Any]:
        """Extract processing configuration details from session state."""
        details = {}
        
        # Extract from processing memory
        if step in st.session_state.processing_memory.get('configurations', {}):
            details = st.session_state.processing_memory['configurations'][step]
        
        return details
    
    def _extract_user_concerns(self, messages: List[Dict]) -> List[str]:
        """Extract user concerns and questions from conversation."""
        concerns = []
        concern_indicators = ['worried', 'concern', 'problem', 'issue', 'help', 'confused']
        
        for msg in messages:
            if msg['role'] == 'user':
                content = msg['content'].lower()
                if any(indicator in content for indicator in concern_indicators):
                    concerns.append(msg['content'])
        
        return concerns[-2:]  # Keep last 2 concerns
    
    def _extract_recommendations(self, messages: List[Dict]) -> List[str]:
        """Extract recommendations given by the assistant."""
        recommendations = []
        
        for msg in messages:
            if msg['role'] == 'assistant':
                content = msg['content']
                if 'recommend' in content.lower() or 'suggest' in content.lower():
                    # Extract recommendation sentences
                    sentences = content.split('.')
                    for sentence in sentences:
                        if 'recommend' in sentence.lower() or 'suggest' in sentence.lower():
                            recommendations.append(sentence.strip())
        
        return recommendations[-3:]  # Keep last 3 recommendations
    
    def _detect_current_step_from_page(self, page: str) -> str:
        """Detect current processing step from page selection."""
        step_mapping = {
            "Data Upload & Schema": "upload",
            "Data Cleaning & Validation": "cleaning", 
            "Weight Application & Statistical Computation": "weighting",
            "Analysis & Visualization": "analysis",
            "Report Generation": "reporting"
        }
        return step_mapping.get(page, "general")
    
    def _build_intelligent_context(self, user_message: str) -> str:
        """Build intelligent, pruned context with memory integration for Gemini."""
        context_parts = []
        
        # Current step and status
        context_parts.append(f"CURRENT STEP: {self.current_step.upper()}")
        
        # Real-time data context
        data_context = self._get_current_data_context()
        if data_context:
            context_parts.append(f"CURRENT DATA CONTEXT:\n{data_context}")
        
        # Cross-step memory integration (pruned)
        memory_context = self._get_pruned_memory_context()
        if memory_context:
            context_parts.append(f"RELEVANT PREVIOUS STEPS:\n{memory_context}")
        
        # Current conversation context (last few exchanges)
        conversation_context = self._get_recent_conversation_context()
        if conversation_context:
            context_parts.append(f"RECENT CONVERSATION:\n{conversation_context}")
        
        # Processing configurations and results
        processing_context = self._get_processing_context()
        if processing_context:
            context_parts.append(f"PROCESSING DETAILS:\n{processing_context}")
        
        return "\n\n".join(context_parts)
    
    def _get_current_data_context(self) -> str:
        """Get current data state and statistics."""
        context = []
        
        if hasattr(st.session_state, 'data') and st.session_state.data is not None:
            data = st.session_state.data
            context.append(f"- Dataset: {len(data):,} rows × {len(data.columns)} columns")
            
            missing_count = data.isnull().sum().sum()
            missing_pct = (missing_count / (len(data) * len(data.columns))) * 100
            context.append(f"- Missing values: {missing_count:,} ({missing_pct:.2f}%)")
            
            # Data types
            numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
            context.append(f"- Column types: {numeric_cols} numeric, {len(data.columns) - numeric_cols} categorical")
            
            # Recent processing
            if hasattr(st.session_state, 'cleaned_data') and st.session_state.cleaned_data is not None:
                cleaned = st.session_state.cleaned_data
                removed = len(data) - len(cleaned)
                context.append(f"- Cleaning: {removed:,} rows removed, {len(cleaned):,} remaining")
        
        return "\n".join(context) if context else ""
    
    def _get_pruned_memory_context(self) -> str:
        """Get pruned cross-step memory to fit token limits."""
        if not st.session_state.step_summaries:
            return ""
        
        memory_parts = []
        # Prioritize recently completed steps
        completed_steps = [(step, summary) for step, summary in st.session_state.step_summaries.items() 
                          if step != self.current_step]
        
        # Sort by relevance and recency
        completed_steps.sort(key=lambda x: x[1]['timestamp'], reverse=True)
        
        for step, summary in completed_steps[:2]:  # Limit to 2 most recent steps
            step_info = []
            step_info.append(f"{step.upper()}:")
            
            if summary.get('data_insights'):
                insights = summary['data_insights']
                if 'rows' in insights:
                    step_info.append(f"  Data: {insights['rows']:,} rows processed")
                if 'rows_removed' in insights and insights['rows_removed'] > 0:
                    step_info.append(f"  Removed: {insights['rows_removed']:,} rows")
            
            if summary.get('processing_details'):
                details = summary['processing_details']
                if details:
                    key_details = list(details.keys())[:2]  # Limit details
                    step_info.append(f"  Methods: {', '.join(key_details)}")
            
            if summary.get('key_topics'):
                topics = summary['key_topics'][:2]  # Limit topics
                step_info.append(f"  Topics: {', '.join(topics)}")
            
            memory_parts.append("\n".join(step_info))
        
        return "\n\n".join(memory_parts)
    
    def _get_recent_conversation_context(self) -> str:
        """Get recent conversation from current step."""
        messages = st.session_state.step_chat_messages.get(self.current_step, [])
        if len(messages) <= 1:
            return ""
        
        # Get last 4 messages (2 exchanges)
        recent = messages[-4:] if len(messages) >= 4 else messages[-2:]
        context_parts = []
        
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Assistant"
            content = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def _get_processing_context(self) -> str:
        """Get current processing configurations and results."""
        context = []
        
        # Current step processing info
        if self.current_step in st.session_state.processing_memory.get('configurations', {}):
            config = st.session_state.processing_memory['configurations'][self.current_step]
            context.append(f"Configuration: {json.dumps(config, indent=2)[:200]}...")
        
        if self.current_step in st.session_state.processing_memory.get('results', {}):
            results = st.session_state.processing_memory['results'][self.current_step]
            context.append(f"Results: {json.dumps(results, indent=2)[:200]}...")
        
        return "\n".join(context) if context else ""
    
    def _create_structured_prompt(self, user_message: str, context: str) -> str:
        """Create structured prompt with system instructions for Gemini."""
        
        # Step-specific instructions
        step_instructions = {
            'upload': "You are helping with data upload and schema configuration. Focus on file formats, column mapping, and data validation.",
            'cleaning': "You are helping with data cleaning and validation. Focus on missing value treatment, outlier detection, and data quality improvement.",
            'weighting': "You are helping with survey weight application and statistical computation. Focus on weighting methodologies and statistical analysis.",
            'analysis': "You are helping with data analysis and visualization. Focus on insights, patterns, and statistical interpretation.",
            'reporting': "You are helping with report generation. Focus on documentation, presentation, and export options."
        }
        
        current_instruction = step_instructions.get(self.current_step, "You are helping with survey data processing.")
        
        system_prompt = f"""You are an expert Survey Data Processing Assistant with deep knowledge of statistical methods and data science.

CURRENT ROLE: {current_instruction}

CONTEXT:
{context}

CORE PRINCIPLES:
1. **ACCURACY**: Provide precise, data-driven responses based on the actual dataset and processing state
2. **REASONING**: Explain your reasoning clearly and show how you arrived at conclusions
3. **CONTEXT-AWARENESS**: Reference specific data points, configurations, and previous steps when relevant
4. **CLARITY**: Ask clarifying questions if the user's request is ambiguous or incomplete
5. **PRACTICAL**: Focus on actionable advice and next steps

RESPONSE GUIDELINES:
- Use actual numbers and specifics from the current dataset
- Reference previous processing steps and their outcomes when relevant
- Provide clear reasoning for recommendations
- Ask follow-up questions to clarify ambiguous requests
- Highlight important considerations and potential issues
- Keep responses focused and actionable

USER QUERY: {user_message}

Provide a helpful, accurate, and contextually relevant response:"""

        return system_prompt
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    def _prune_context_for_tokens(self, context: str) -> str:
        """Prune context to fit within token limits."""
        estimated_tokens = self._estimate_token_count(context)
        
        if estimated_tokens <= self.available_tokens:
            return context
        
        # Prune context sections by priority
        context_sections = context.split('\n\n')
        pruned_sections = []
        current_tokens = 0
        
        # Priority order: Current data, Recent conversation, Processing details, Memory
        section_priorities = {
            'CURRENT DATA CONTEXT': 1,
            'RECENT CONVERSATION': 2, 
            'PROCESSING DETAILS': 3,
            'RELEVANT PREVIOUS STEPS': 4
        }
        
        # Sort sections by priority
        sections_with_priority = []
        for section in context_sections:
            priority = 5  # Default low priority
            for key, p in section_priorities.items():
                if key in section:
                    priority = p
                    break
            sections_with_priority.append((priority, section))
        
        sections_with_priority.sort(key=lambda x: x[0])
        
        # Add sections until token limit
        for priority, section in sections_with_priority:
            section_tokens = self._estimate_token_count(section)
            if current_tokens + section_tokens <= self.available_tokens * 0.8:  # 80% buffer
                pruned_sections.append(section)
                current_tokens += section_tokens
            else:
                # Try to include truncated version
                max_chars = (self.available_tokens - current_tokens) * 4
                if max_chars > 100:  # Minimum useful size
                    truncated = section[:max_chars] + "...[truncated]"
                    pruned_sections.append(truncated)
                break
        
        pruned_context = '\n\n'.join(pruned_sections)
        
        # Log pruning for optimization
        st.session_state.context_pruning_log.append({
            'timestamp': datetime.now().isoformat(),
            'original_tokens': estimated_tokens,
            'pruned_tokens': self._estimate_token_count(pruned_context),
            'step': self.current_step
        })
        
        return pruned_context
    
    def _load_fallback_responses(self) -> Dict[str, List[str]]:
        """Load fallback responses for when API is unavailable."""
        return {
            'general': [
                "I'm currently experiencing connectivity issues. Based on your question, I'd recommend checking the current data statistics in the sidebar.",
                "The AI service is temporarily unavailable. You can continue with the data processing steps, and I'll be back shortly to help.",
                "I'm having trouble connecting to the AI service right now. Try refreshing the page or check the help documentation."
            ],
            'data_quality': [
                "Without AI access, I can't analyze specifics, but you can review data quality metrics in the data overview section.",
                "Check the data preview table for missing values and outliers while I work on reconnecting.",
                "The data quality assessment tools are available in the cleaning section to help identify issues."
            ],
            'processing': [
                "I can't provide detailed guidance right now, but the processing steps have built-in validation to help guide you.",
                "Each processing step includes recommendations and best practices to help you proceed safely.",
                "While I'm offline, you can use the step-by-step guides in each section."
            ]
        }
    
    def _get_fallback_response(self, user_message: str) -> str:
        """Get appropriate fallback response when API is unavailable."""
        message_lower = user_message.lower()
        
        # Categorize the message
        if any(word in message_lower for word in ['missing', 'null', 'quality', 'clean']):
            category = 'data_quality'
        elif any(word in message_lower for word in ['process', 'step', 'how', 'method']):
            category = 'processing'
        else:
            category = 'general'
        
        # Get random fallback from category
        import random
        fallbacks = self._fallback_responses.get(category, self._fallback_responses['general'])
        return random.choice(fallbacks)
    
    def _validate_user_query(self, user_message: str) -> tuple[bool, str]:
        """Validate user query and suggest clarifications if needed."""
        message_lower = user_message.lower()
        
        # Check for too vague queries
        vague_indicators = ['help', 'what', 'how', 'why', 'tell me', 'explain']
        specific_indicators = ['missing values', 'outliers', 'columns', 'rows', 'clean', 'weight', 'analysis']
        
        is_vague = (any(indicator in message_lower for indicator in vague_indicators) and
                   not any(indicator in message_lower for indicator in specific_indicators) and
                   len(user_message.split()) < 4)
        
        if is_vague:
            clarification = self._suggest_clarification(user_message)
            return False, clarification
        
        return True, ""
    
    def _suggest_clarification(self, user_message: str) -> str:
        """Suggest clarifying questions for vague queries."""
        message_lower = user_message.lower()
        
        suggestions = {
            'help': "I'd be happy to help! Could you be more specific about what you need assistance with? For example:\n• Data upload issues\n• Cleaning specific columns\n• Understanding analysis results",
            'what': "Could you specify what aspect you'd like to know about? For example:\n• What data quality issues exist?\n• What cleaning methods to use?\n• What the analysis results mean?",
            'how': "I can guide you through the process! What specifically would you like to know how to do?\n• How to handle missing values?\n• How to apply survey weights?\n• How to interpret results?"
        }
        
        for key, suggestion in suggestions.items():
            if key in message_lower:
                return suggestion
        
        return "Could you provide more details about what you'd like to know? I can help with specific questions about your data, processing steps, or analysis results."
    
    def generate_response(self, user_message: str) -> str:
        """Generate intelligent response using Gemini with comprehensive context and fallback handling."""
        try:
            # Validate query first
            is_valid, clarification = self._validate_user_query(user_message)
            if not is_valid:
                return clarification
            
            # Check if client is available
            if not self.client or not self.model:
                return self._get_fallback_response(user_message)
            
            # Build and prune context
            context = self._build_intelligent_context(user_message)
            pruned_context = self._prune_context_for_tokens(context)
            
            # Create structured prompt
            prompt = self._create_structured_prompt(user_message, pruned_context)
            
            # Generate response with Gemini
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                # Store interaction in memory
                self._store_interaction(user_message, response.text)
                return response.text
            else:
                return self._get_fallback_response(user_message)
                
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return self._get_fallback_response(user_message)
    
    def _store_interaction(self, user_message: str, ai_response: str):
        """Store interaction for memory and future context."""
        timestamp = datetime.now().isoformat()
        
        # Update processing memory with insights from interaction
        if 'insights' not in st.session_state.processing_memory:
            st.session_state.processing_memory['insights'] = {}
        
        # Extract and store insights
        insights = {
            'timestamp': timestamp,
            'user_question_type': self._categorize_question(user_message),
            'response_category': self._categorize_response(ai_response),
            'step': self.current_step
        }
        
        step_insights = st.session_state.processing_memory['insights'].get(self.current_step, [])
        step_insights.append(insights)
        st.session_state.processing_memory['insights'][self.current_step] = step_insights[-10:]  # Keep last 10
    
    def _categorize_question(self, question: str) -> str:
        """Categorize user question for insights tracking."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['missing', 'null', 'empty']):
            return 'missing_data'
        elif any(word in question_lower for word in ['outlier', 'anomaly', 'unusual']):
            return 'outliers'
        elif any(word in question_lower for word in ['clean', 'process', 'method']):
            return 'methodology'
        elif any(word in question_lower for word in ['why', 'explain', 'reason']):
            return 'explanation'
        elif any(word in question_lower for word in ['recommend', 'suggest', 'should']):
            return 'recommendation'
        else:
            return 'general_inquiry'
    
    def _categorize_response(self, response: str) -> str:
        """Categorize AI response for insights tracking."""
        response_lower = response.lower()
        
        if any(word in response_lower for word in ['recommend', 'suggest']):
            return 'recommendation'
        elif any(word in response_lower for word in ['because', 'due to', 'reason']):
            return 'explanation'
        elif any(word in response_lower for word in ['data shows', 'analysis indicates']):
            return 'data_analysis'
        else:
            return 'information'
    
    def display_chatbot(self):
        """Display the enhanced Gemini-powered chatbot interface."""
        # Enhanced CSS for better UI
        st.markdown("""
        <style>
        .gemini-chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .memory-indicator {
            background: rgba(255, 255, 255, 0.1);
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        .step-context {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Update current step
        self.current_step = self._detect_current_step_from_page(self.current_page)
        
        # Initialize step messages
        if self.current_step not in st.session_state.step_chat_messages:
            st.session_state.step_chat_messages[self.current_step] = []
        
        # Step names mapping
        step_names = {
            'upload': '📁 Data Upload & Schema',
            'cleaning': '🧹 Data Cleaning & Validation', 
            'weighting': '⚖️ Weight Application & Analysis',
            'analysis': '📈 Analysis & Visualization',
            'reporting': '📄 Report Generation'
        }
        
        current_step_name = step_names.get(self.current_step, self.current_step.title())
        
        # Memory status indicator
        memory_count = len(st.session_state.step_summaries)
        memory_indicator = f"🧠 {memory_count} steps remembered" if memory_count > 0 else "🆕 Fresh start"
        
        # Enhanced header
        st.markdown(f"""
        <div class="gemini-chat-header">
            <h3 style="margin: 0; display: flex; align-items: center; justify-content: center;">
                🤖 AI Assistant - {current_step_name}
                <span class="memory-indicator">{memory_indicator}</span>
                <span style="margin-left: auto; font-size: 0.7em; opacity: 0.9;">✨ Powered by Gemini</span>
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # API key check and setup
        if not self.client:
            self._display_api_key_setup()
            return
        
        # Step context indicator
        data_context = self._get_current_data_context()
        if data_context:
            st.markdown(f"""
            <div class="step-context">
                <strong>📊 Current Context:</strong><br>
                {data_context.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
        
        # Welcome message for new steps
        messages = st.session_state.step_chat_messages[self.current_step]
        if not messages:
            welcome_message = self._get_step_welcome_message()
            st.session_state.step_chat_messages[self.current_step].append({
                'role': 'assistant',
                'content': welcome_message,
                'timestamp': datetime.now().isoformat()
            })
        
        # Display chat messages
        for message in st.session_state.step_chat_messages[self.current_step]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input(f"Ask me about {current_step_name.lower()}..."):
            # Add user message
            st.session_state.step_chat_messages[self.current_step].append({
                'role': 'user',
                'content': prompt,
                'timestamp': datetime.now().isoformat()
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display AI response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing your question..."):
                    response = self.generate_response(prompt)
                st.markdown(response)
            
            # Add AI response to messages
            st.session_state.step_chat_messages[self.current_step].append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Rerun to show new messages
            st.rerun()
        
        # Sidebar with additional features
        self._display_sidebar_features()
    
    def _display_api_key_setup(self):
        """Display API key setup interface."""
        st.warning("🔑 Gemini API Key Required")
        st.markdown("""
        To use the AI chatbot, you need to configure your Google Gemini API key:
        
        1. **Get your API key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. **Set environment variable**: Add `GEMINI_API_KEY` to your environment
        3. **Restart the application** to activate the chatbot
        
        The chatbot provides intelligent, context-aware assistance across all processing steps.
        """)
        
        # Fallback options
        st.info("💡 **Alternative**: You can continue without the chatbot - all processing features remain fully functional.")
    
    def _get_step_welcome_message(self) -> str:
        """Get contextual welcome message for current step."""
        data_context = ""
        if hasattr(st.session_state, 'data') and st.session_state.data is not None:
            data = st.session_state.data
            data_context = f" I can see you have {len(data):,} rows and {len(data.columns)} columns loaded."
        
        messages = {
            'upload': f"👋 Welcome! I'm your AI assistant for data upload and schema configuration.{data_context} I can help you understand your data structure, validate uploads, and configure schemas. What would you like to explore?",
            'cleaning': f"🧹 Hello! I'm here to assist with data cleaning and validation.{data_context} I can help you identify data quality issues, recommend cleaning methods, and guide you through the validation process. What cleaning challenges are you facing?",
            'weighting': f"⚖️ Hi there! I'm your expert assistant for survey weighting and statistical computation.{data_context} I can help you understand weighting methodologies, apply appropriate techniques, and interpret statistical results. What weighting questions do you have?",
            'analysis': f"📈 Welcome to the analysis phase!{data_context} I can help you explore patterns, understand statistical results, create meaningful visualizations, and interpret your findings. What insights are you looking for?",
            'reporting': f"📄 Ready to generate reports!{data_context} I can guide you through report creation, suggest appropriate sections, help format findings, and ensure comprehensive documentation. What kind of report are you creating?"
        }
        
        return messages.get(self.current_step, f"👋 Hello! I'm your AI assistant for survey data processing. How can I help you today?")
    
    def _display_sidebar_features(self):
        """Display additional chatbot features in sidebar."""
        with st.sidebar:
            st.markdown("### 🤖 Chatbot Features")
            
            # Memory status
            if st.session_state.step_summaries:
                st.markdown("**📚 Memory Status:**")
                for step, summary in st.session_state.step_summaries.items():
                    topics = summary.get('key_topics', [])
                    st.markdown(f"• {step.title()}: {len(topics)} topics remembered")
            
            # Token usage info
            if st.session_state.context_pruning_log:
                recent_pruning = st.session_state.context_pruning_log[-1]
                efficiency = (recent_pruning['pruned_tokens'] / recent_pruning['original_tokens']) * 100
                st.markdown(f"**⚡ Token Efficiency:** {efficiency:.1f}%")
            
            # Export chat
            if st.button("💾 Export Chat History"):
                self._export_chat_history()
            
            # Clear step memory
            if st.button("🗑️ Clear Current Step"):
                if self.current_step in st.session_state.step_chat_messages:
                    st.session_state.step_chat_messages[self.current_step] = []
                    st.rerun()
    
    def _export_chat_history(self):
        """Export comprehensive chat history with context."""
        if not st.session_state.step_chat_messages:
            st.warning("No chat history to export")
            return
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'step_conversations': st.session_state.step_chat_messages,
            'step_summaries': st.session_state.step_summaries,
            'processing_memory': st.session_state.processing_memory,
            'context_pruning_log': st.session_state.context_pruning_log
        }
        
        export_json = json.dumps(export_data, indent=2, default=str)
        
        st.download_button(
            label="📥 Download Complete History",
            data=export_json,
            file_name=f"gemini_chat_history_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
    
    def update_processing_memory(self, step: str, config_type: str, data: Dict[str, Any]):
        """Update processing memory with new configuration or results."""
        if config_type not in st.session_state.processing_memory:
            st.session_state.processing_memory[config_type] = {}
        
        st.session_state.processing_memory[config_type][step] = {
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_context_for_step(self, step: str) -> Dict[str, Any]:
        """Get comprehensive context for a specific step."""
        context = {
            'step': step,
            'messages': st.session_state.step_chat_messages.get(step, []),
            'summary': st.session_state.step_summaries.get(step, {}),
            'memory': st.session_state.cross_step_memory.get(step, {})
        }
        return context
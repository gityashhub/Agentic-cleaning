import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

class Visualizations:
    """Create interactive visualizations for survey data analysis."""
    
    def __init__(self):
        # Set default color palette
        self.color_palette = px.colors.qualitative.Set2
        
    def plot_missing_patterns(self, df):
        """Create a visualization of missing data patterns."""
        # Calculate missing data percentages
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        # Create bar chart of missing data by column
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=missing_percent.index,
            y=missing_percent.values,
            name='Missing Data %',
            marker_color='lightcoral',
            text=[f'{x:.1f}%' for x in missing_percent.values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Missing Data Patterns by Variable',
            xaxis_title='Variables',
            yaxis_title='Missing Data Percentage',
            showlegend=False,
            height=500
        )
        
        # Rotate x-axis labels if there are many variables
        if len(missing_percent) > 10:
            fig.update_xaxes(tickangle=45)
        
        return fig
    
    def plot_distribution(self, df, column):
        """Create distribution plot for a numeric variable."""
        if column not in df.columns:
            return None
        
        data = df[column].dropna()
        
        if len(data) == 0:
            return None
        
        # Create subplot with histogram and box plot
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'Distribution of {column}', 'Box Plot'),
            vertical_spacing=0.1
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=50,
                name='Frequency',
                marker_color='skyblue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                x=data,
                name=column,
                marker_color='lightgreen',
                boxpoints='outliers'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text=f'Statistical Distribution: {column}'
        )
        
        return fig
    
    def plot_correlation_matrix(self, df, variables=None):
        """Create correlation matrix heatmap for numeric variables."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if variables:
            numeric_df = numeric_df[variables]
        
        if numeric_df.empty:
            return None
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Correlation Matrix',
            height=600,
            width=600
        )
        
        return fig
    
    def plot_categorical_distribution(self, df, column, max_categories=20):
        """Create bar chart for categorical variable distribution."""
        if column not in df.columns:
            return None
        
        # Get value counts
        value_counts = df[column].value_counts().head(max_categories)
        
        if len(value_counts) == 0:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            name='Count',
            marker_color='lightblue',
            text=value_counts.values,
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f'Distribution of {column}',
            xaxis_title=column,
            yaxis_title='Count',
            showlegend=False,
            height=500
        )
        
        # Rotate labels if needed
        if len(value_counts) > 10:
            fig.update_xaxes(tickangle=45)
        
        return fig
    
    def plot_weighted_vs_unweighted(self, df, variable, weight_column):
        """Compare weighted vs unweighted distributions."""
        if variable not in df.columns or weight_column not in df.columns:
            return None
        
        data = df[[variable, weight_column]].dropna()
        
        if len(data) == 0:
            return None
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Unweighted Distribution', 'Weighted Distribution'),
            horizontal_spacing=0.1
        )
        
        # Unweighted histogram
        fig.add_trace(
            go.Histogram(
                x=data[variable],
                nbinsx=30,
                name='Unweighted',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Weighted histogram (approximate using sample weights)
        # This is a simplified approach - in practice, you'd want more sophisticated weighting
        weights = data[weight_column]
        weighted_data = np.repeat(data[variable], np.maximum(1, (weights * 10).astype(int)))
        
        fig.add_trace(
            go.Histogram(
                x=weighted_data,
                nbinsx=30,
                name='Weighted',
                marker_color='lightcoral',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'Weighted vs Unweighted: {variable}',
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_survey_weights_diagnostics(self, weights):
        """Create diagnostic plots for survey weights."""
        if len(weights) == 0:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Weight Distribution',
                'Weight Box Plot', 
                'Weight vs Observation Order',
                'Weight Summary Statistics'
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "table"}]]
        )
        
        # Weight distribution histogram
        fig.add_trace(
            go.Histogram(
                x=weights,
                nbinsx=50,
                name='Weight Distribution',
                marker_color='lightgreen'
            ),
            row=1, col=1
        )
        
        # Weight box plot
        fig.add_trace(
            go.Box(
                y=weights,
                name='Weights',
                marker_color='orange',
                boxpoints='outliers'
            ),
            row=1, col=2
        )
        
        # Weight vs observation order
        fig.add_trace(
            go.Scatter(
                x=list(range(len(weights))),
                y=weights,
                mode='markers',
                name='Weight Order',
                marker=dict(color='purple', size=3)
            ),
            row=2, col=1
        )
        
        # Summary statistics table
        stats = {
            'Statistic': ['Count', 'Mean', 'Std', 'Min', 'Max', 'CV'],
            'Value': [
                f'{len(weights):,}',
                f'{weights.mean():.3f}',
                f'{weights.std():.3f}',
                f'{weights.min():.3f}',
                f'{weights.max():.3f}',
                f'{weights.std()/weights.mean():.3f}'
            ]
        }
        
        fig.add_trace(
            go.Table(
                header=dict(values=list(stats.keys()),
                           fill_color='lightblue',
                           align='left'),
                cells=dict(values=list(stats.values()),
                          fill_color='white',
                          align='left')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Survey Weights Diagnostic Plots',
            showlegend=False,
            height=800
        )
        
        return fig
    
    def plot_stratified_results(self, stratified_results, variable, statistic='weighted_mean'):
        """Plot stratified analysis results."""
        if not stratified_results:
            return None
        
        # Extract data for plotting
        strata = []
        values = []
        errors = []
        
        for stratum, results in stratified_results.items():
            if variable in results:
                var_results = results[variable]
                if isinstance(var_results, dict) and statistic in var_results:
                    strata.append(stratum)
                    values.append(var_results[statistic])
                    # Add error bars if available
                    if 'std_error' in var_results:
                        errors.append(var_results['std_error'])
                    else:
                        errors.append(0)
        
        if not strata:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=strata,
            y=values,
            error_y=dict(type='data', array=errors),
            name=f'{statistic}',
            marker_color='lightcoral',
            text=[f'{v:.3f}' for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f'Stratified Analysis: {variable} ({statistic})',
            xaxis_title='Strata',
            yaxis_title=statistic.replace('_', ' ').title(),
            showlegend=False,
            height=500
        )
        
        if len(strata) > 8:
            fig.update_xaxes(tickangle=45)
        
        return fig
    
    def plot_margin_of_error_comparison(self, margins_df):
        """Create comparison plot of margins of error across variables."""
        if margins_df.empty:
            return None
        
        # Extract margin of error values
        variables = margins_df.index.tolist()
        margins = margins_df['Margin of Error'].tolist()
        
        # Remove any NaN values
        valid_data = [(var, margin) for var, margin in zip(variables, margins) 
                     if not pd.isna(margin)]
        
        if not valid_data:
            return None
        
        variables, margins = zip(*valid_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(variables),
            y=list(margins),
            name='Margin of Error',
            marker_color='gold',
            text=[f'{m:.4f}' for m in margins],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Margins of Error by Variable',
            xaxis_title='Variables',
            yaxis_title='Margin of Error',
            showlegend=False,
            height=500
        )
        
        if len(variables) > 10:
            fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_data_quality_dashboard(self, df, cleaning_report=None):
        """Create a comprehensive data quality dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Missing Data by Variable',
                'Data Types Distribution',
                'Numeric Variables Summary',
                'Data Quality Score'
            ),
            specs=[[{"type": "xy"}, {"type": "domain"}],
                   [{"type": "table"}, {"type": "indicator"}]]
        )
        
        # Missing data plot
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        fig.add_trace(
            go.Bar(
                x=missing_percent.index[:10],  # Top 10 variables
                y=missing_percent.values[:10],
                name='Missing %',
                marker_color='red'
            ),
            row=1, col=1
        )
        
        # Data types pie chart
        dtypes_count = df.dtypes.value_counts()
        fig.add_trace(
            go.Pie(
                labels=[str(dtype) for dtype in dtypes_count.index],
                values=dtypes_count.values,
                name="Data Types"
            ),
            row=1, col=2
        )
        
        # Numeric summary table
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            summary_stats = numeric_df.describe().round(3)
            
            # Convert to table format
            table_data = {
                'Variable': summary_stats.columns[:5].tolist(),  # First 5 variables
                'Mean': [f"{summary_stats.loc['mean', col]:.3f}" for col in summary_stats.columns[:5]],
                'Std': [f"{summary_stats.loc['std', col]:.3f}" for col in summary_stats.columns[:5]],
                'Min': [f"{summary_stats.loc['min', col]:.3f}" for col in summary_stats.columns[:5]],
                'Max': [f"{summary_stats.loc['max', col]:.3f}" for col in summary_stats.columns[:5]]
            }
            
            fig.add_trace(
                go.Table(
                    header=dict(values=list(table_data.keys()),
                               fill_color='lightblue'),
                    cells=dict(values=list(table_data.values()),
                              fill_color='white')
                ),
                row=2, col=1
            )
        
        # Data quality score (simplified calculation)
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_score = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=completeness_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Data Completeness (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Data Quality Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig

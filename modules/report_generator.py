import pandas as pd
import numpy as np
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart

class ReportGenerator:
    """Generate comprehensive statistical reports in various formats."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Normal'],
            fontSize=16,
            spaceAfter=20,
            alignment=1,
            textColor=colors.grey
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.darkblue
        ))
    
    def generate_report(self, cleaned_data, weighted_results, processing_log, config):
        """
        Generate a comprehensive statistical report.
        
        Args:
            cleaned_data: DataFrame with cleaned survey data
            weighted_results: Dict with statistical results
            processing_log: List of processing steps
            config: Report configuration
            
        Returns:
            dict: Report data including preview and file content
        """
        report_data = {
            'config': config,
            'generated_at': datetime.now(),
            'data_summary': self._get_data_summary(cleaned_data),
            'results_summary': self._get_results_summary(weighted_results)
        }
        
        if config.get('format', 'PDF') == 'PDF':
            pdf_content = self._generate_pdf_report(
                cleaned_data, weighted_results, processing_log, config, report_data
            )
            preview = self._generate_text_preview(report_data)
            
            return {
                'pdf_content': pdf_content,
                'preview': preview,
                'metadata': report_data
            }
        else:
            # HTML format
            html_content = self._generate_html_report(
                cleaned_data, weighted_results, processing_log, config, report_data
            )
            preview = self._generate_text_preview(report_data)
            
            return {
                'html_content': html_content,
                'preview': preview,
                'metadata': report_data
            }
    
    def _generate_pdf_report(self, cleaned_data, weighted_results, processing_log, config, report_data):
        """Generate PDF report using ReportLab."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        # Title page
        story.extend(self._create_title_page(config, report_data))
        story.append(PageBreak())
        
        # Executive summary
        if 'Executive Summary' in config.get('custom_sections', []) or \
           config.get('template_style') == 'Executive Summary':
            story.extend(self._create_executive_summary(weighted_results))
            story.append(PageBreak())
        
        # Data overview
        story.extend(self._create_data_overview(cleaned_data, report_data))
        
        # Methodology section
        if config.get('include_methodology', True):
            story.extend(self._create_methodology_section(processing_log))
        
        # Results section
        story.extend(self._create_results_section(weighted_results))
        
        # Data quality diagnostics
        if config.get('include_diagnostics', True):
            story.extend(self._create_diagnostics_section(cleaned_data, processing_log))
        
        # Custom sections
        for section in config.get('custom_sections', []):
            if section not in ['Executive Summary']:
                story.extend(self._create_custom_section(section, weighted_results))
        
        # Build PDF
        try:
            doc.build(story)
            buffer.seek(0)
            pdf_content = buffer.getvalue()
            
            # Verify PDF content is not empty
            if len(pdf_content) == 0:
                raise ValueError("Generated PDF is empty")
                
            return pdf_content
        except Exception as e:
            # If PDF generation fails, create a simple error PDF
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            error_story = [
                Paragraph("Report Generation Error", self.styles['Title']),
                Spacer(1, 12),
                Paragraph(f"An error occurred while generating the report: {str(e)}", self.styles['Normal']),
                Spacer(1, 12),
                Paragraph("Please try regenerating the report or contact support.", self.styles['Normal'])
            ]
            doc.build(error_story)
            buffer.seek(0)
            return buffer.getvalue()
    
    def _create_title_page(self, config, report_data):
        """Create the title page content."""
        story = []
        
        # Title
        title = Paragraph(config.get('title', 'Survey Data Analysis Report'), 
                         self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Subtitle
        subtitle = Paragraph(config.get('subtitle', 'Statistical Analysis Report'), 
                           self.styles['CustomSubtitle'])
        story.append(subtitle)
        story.append(Spacer(1, 30))
        
        # Metadata table
        metadata = [
            ['Author:', config.get('author', 'Statistical Agency')],
            ['Organization:', config.get('organization', 'National Statistical Office')],
            ['Generated:', report_data['generated_at'].strftime('%Y-%m-%d %H:%M:%S')],
            ['Data Points:', f"{report_data['data_summary']['total_observations']:,}"],
            ['Variables:', f"{report_data['data_summary']['total_variables']:,}"]
        ]
        
        metadata_table = Table(metadata, colWidths=[2*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        story.append(metadata_table)
        
        return story
    
    def _create_executive_summary(self, weighted_results):
        """Create executive summary section."""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Key findings
        summary_text = """
        This report presents the results of comprehensive survey data analysis including 
        data cleaning, validation, and weighted statistical computations. The analysis 
        ensures methodological consistency and provides reliable estimates for official 
        statistical releases.
        """
        
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Key statistics
        if 'summary_stats' in weighted_results and not weighted_results['summary_stats'].empty:
            stats_df = weighted_results['summary_stats']
            
            key_stats = []
            for var in stats_df.index[:5]:  # Top 5 variables
                if 'weighted_mean' in stats_df.columns:
                    mean_val = stats_df.loc[var, 'weighted_mean']
                    if not pd.isna(mean_val):
                        key_stats.append([var, f"{mean_val:.3f}"])
            
            if key_stats:
                story.append(Paragraph("Key Estimates:", self.styles['Heading2']))
                stats_table = Table([['Variable', 'Weighted Mean']] + key_stats)
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(stats_table)
        
        return story
    
    def _create_data_overview(self, cleaned_data, report_data):
        """Create data overview section."""
        story = []
        
        story.append(Paragraph("Data Overview", self.styles['SectionHeader']))
        
        # Data summary
        summary = report_data['data_summary']
        overview_text = f"""
        The dataset contains {summary['total_observations']:,} observations across 
        {summary['total_variables']} variables. Data completeness is 
        {summary['completeness_rate']:.1%}, with {summary['numeric_variables']} 
        numeric variables and {summary['categorical_variables']} categorical variables.
        """
        
        story.append(Paragraph(overview_text, self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Variable summary table
        var_info = []
        for col in cleaned_data.columns[:10]:  # First 10 columns
            dtype = str(cleaned_data[col].dtype)
            missing_pct = (cleaned_data[col].isnull().sum() / len(cleaned_data)) * 100
            var_info.append([col, dtype, f"{missing_pct:.1f}%"])
        
        if var_info:
            story.append(Paragraph("Variable Summary:", self.styles['Heading2']))
            var_table = Table([['Variable', 'Type', 'Missing %']] + var_info)
            var_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(var_table)
        
        story.append(Spacer(1, 20))
        return story
    
    def _create_methodology_section(self, processing_log):
        """Create methodology section."""
        story = []
        
        story.append(Paragraph("Methodology", self.styles['SectionHeader']))
        
        methodology_text = """
        This analysis follows standard survey methodology practices for official statistics. 
        The processing workflow includes data validation, cleaning, outlier detection, 
        and weighted estimation procedures.
        """
        
        story.append(Paragraph(methodology_text, self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Processing steps
        if processing_log:
            story.append(Paragraph("Processing Steps:", self.styles['Heading2']))
            
            for i, step in enumerate(processing_log, 1):
                step_text = f"{i}. {step['step']} - {step['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
                story.append(Paragraph(step_text, self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _create_results_section(self, weighted_results):
        """Create results section."""
        story = []
        
        story.append(Paragraph("Statistical Results", self.styles['SectionHeader']))
        
        # Summary statistics
        if 'summary_stats' in weighted_results and not weighted_results['summary_stats'].empty:
            story.append(Paragraph("Summary Statistics", self.styles['Heading2']))
            
            stats_df = weighted_results['summary_stats']
            
            # Convert DataFrame to table format for PDF
            table_data = [list(stats_df.columns)]
            for idx in stats_df.index:
                row = [str(idx)]
                for col in stats_df.columns:
                    value = stats_df.loc[idx, col]
                    if pd.isna(value):
                        row.append('N/A')
                    elif isinstance(value, (int, float)):
                        row.append(f"{value:.3f}")
                    else:
                        row.append(str(value))
                table_data.append(row)
            
            if len(table_data) > 1:
                results_table = Table(table_data)
                results_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(results_table)
        
        # Margins of error
        if 'margins_of_error' in weighted_results and not weighted_results['margins_of_error'].empty:
            story.append(Spacer(1, 20))
            story.append(Paragraph("Margins of Error", self.styles['Heading2']))
            
            moe_df = weighted_results['margins_of_error']
            moe_data = [['Variable', 'Margin of Error']]
            
            for idx in moe_df.index:
                moe_value = moe_df.loc[idx, 'Margin of Error']
                if not pd.isna(moe_value):
                    moe_data.append([str(idx), f"{moe_value:.4f}"])
            
            if len(moe_data) > 1:
                moe_table = Table(moe_data)
                moe_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(moe_table)
        
        story.append(Spacer(1, 20))
        return story
    
    def _create_diagnostics_section(self, cleaned_data, processing_log):
        """Create data quality diagnostics section."""
        story = []
        
        story.append(Paragraph("Data Quality Diagnostics", self.styles['SectionHeader']))
        
        # Cleaning summary
        cleaning_summary = "Data quality assessment and cleaning procedures were applied "
        cleaning_summary += "to ensure data reliability and methodological consistency."
        
        story.append(Paragraph(cleaning_summary, self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Processing log summary
        if processing_log:
            story.append(Paragraph("Quality Assurance Steps:", self.styles['Heading2']))
            
            for step in processing_log:
                if 'report' in step:
                    report = step['report']
                    if 'steps_performed' in report:
                        for performed_step in report['steps_performed']:
                            story.append(Paragraph(f"• {performed_step}", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _create_custom_section(self, section_name, weighted_results):
        """Create a custom section."""
        story = []
        
        story.append(Paragraph(section_name, self.styles['SectionHeader']))
        
        if section_name == "Key Findings":
            findings_text = """
            Key findings from the survey analysis will be highlighted in this section.
            Statistical significance and practical implications of the results should be
            discussed with appropriate context for policy and decision-making.
            """
        elif section_name == "Recommendations":
            findings_text = """
            Based on the analysis results, methodological recommendations and suggestions
            for future data collection efforts can be provided in this section.
            """
        else:
            findings_text = f"Content for {section_name} section would be included here."
        
        story.append(Paragraph(findings_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        return story
    
    def _get_data_summary(self, cleaned_data):
        """Get summary statistics about the dataset."""
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        categorical_cols = cleaned_data.select_dtypes(include=['object', 'category']).columns
        
        total_cells = cleaned_data.size
        missing_cells = cleaned_data.isnull().sum().sum()
        
        return {
            'total_observations': len(cleaned_data),
            'total_variables': len(cleaned_data.columns),
            'numeric_variables': len(numeric_cols),
            'categorical_variables': len(categorical_cols),
            'completeness_rate': 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        }
    
    def _get_results_summary(self, weighted_results):
        """Get summary of the statistical results."""
        summary = {
            'has_summary_stats': 'summary_stats' in weighted_results,
            'has_margins_of_error': 'margins_of_error' in weighted_results,
            'has_stratified_results': 'stratified_results' in weighted_results,
            'variables_analyzed': 0
        }
        
        if summary['has_summary_stats'] and not weighted_results['summary_stats'].empty:
            summary['variables_analyzed'] = len(weighted_results['summary_stats'])
        
        return summary
    
    def _generate_text_preview(self, report_data):
        """Generate a text preview of the report."""
        preview = f"""
# {report_data['config'].get('title', 'Survey Data Analysis Report')}

**Generated:** {report_data['generated_at'].strftime('%Y-%m-%d %H:%M:%S')}
**Author:** {report_data['config'].get('author', 'Statistical Agency')}

## Data Summary
- **Observations:** {report_data['data_summary']['total_observations']:,}
- **Variables:** {report_data['data_summary']['total_variables']:,}
- **Completeness:** {report_data['data_summary']['completeness_rate']:.1%}

## Analysis Results
- **Variables Analyzed:** {report_data['results_summary']['variables_analyzed']}
- **Summary Statistics:** {'✓' if report_data['results_summary']['has_summary_stats'] else '✗'}
- **Margins of Error:** {'✓' if report_data['results_summary']['has_margins_of_error'] else '✗'}
- **Stratified Results:** {'✓' if report_data['results_summary']['has_stratified_results'] else '✗'}

The complete report includes detailed methodology, statistical results, and data quality diagnostics.
        """
        
        return preview.strip()
    
    def _generate_html_report(self, cleaned_data, weighted_results, processing_log, config, report_data):
        """Generate HTML report (simplified version)."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{config.get('title', 'Survey Report')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{config.get('title', 'Survey Data Analysis Report')}</h1>
            <p><strong>Generated:</strong> {report_data['generated_at'].strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Author:</strong> {config.get('author', 'Statistical Agency')}</p>
            
            <h2>Data Overview</h2>
            <p>Observations: {report_data['data_summary']['total_observations']:,}</p>
            <p>Variables: {report_data['data_summary']['total_variables']:,}</p>
            <p>Completeness: {report_data['data_summary']['completeness_rate']:.1%}</p>
            
            <h2>Statistical Results</h2>
            <p>Results summary would be displayed here in tabular format.</p>
        </body>
        </html>
        """
        
        return html_content

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import io

class StandardReportTemplate:
    """Standard template for survey analysis reports."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self.page_width = A4[0]
        self.page_height = A4[1]
        self.margin = 72  # 1 inch margin
    
    def _setup_custom_styles(self):
        """Setup custom styles for different report elements."""
        
        # Title styles
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=20,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.grey,
            fontName='Helvetica'
        ))
        
        # Section headers
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.darkblue,
            borderPadding=8
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        ))
        
        # Body text styles
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceBefore=6,
            spaceAfter=6,
            fontName='Helvetica'
        ))
        
        self.styles.add(ParagraphStyle(
            name='BulletText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceBefore=3,
            spaceAfter=3,
            leftIndent=20,
            bulletIndent=10,
            fontName='Helvetica'
        ))
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))
        
        # Methodology box style
        self.styles.add(ParagraphStyle(
            name='MethodologyBox',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceBefore=10,
            spaceAfter=10,
            leftIndent=20,
            rightIndent=20,
            borderWidth=1,
            borderColor=colors.lightgrey,
            borderPadding=10,
            backColor=colors.lightgrey,
            fontName='Helvetica'
        ))
    
    def create_title_page(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> List:
        """Create the title page content."""
        story = []
        
        # Add some space from top
        story.append(Spacer(1, 100))
        
        # Main title
        title = config.get('title', 'Survey Data Analysis Report')
        story.append(Paragraph(title, self.styles['ReportTitle']))
        
        # Subtitle
        subtitle = config.get('subtitle', 'Statistical Analysis and Results')
        story.append(Paragraph(subtitle, self.styles['ReportSubtitle']))
        
        story.append(Spacer(1, 50))
        
        # Organization logo placeholder (if needed)
        # story.append(self._create_logo_placeholder())
        
        story.append(Spacer(1, 30))
        
        # Metadata table
        metadata_info = [
            ['Prepared by:', config.get('author', 'Statistical Agency')],
            ['Organization:', config.get('organization', 'National Statistical Office')],
            ['Report Date:', metadata.get('generated_at', datetime.now()).strftime('%B %d, %Y')],
            ['Dataset Size:', f"{metadata.get('data_summary', {}).get('total_observations', 'N/A'):,} observations"],
            ['Variables Analyzed:', f"{metadata.get('data_summary', {}).get('total_variables', 'N/A')} variables"],
            ['Data Completeness:', f"{metadata.get('data_summary', {}).get('completeness_rate', 0):.1%}"]
        ]
        
        metadata_table = Table(metadata_info, colWidths=[2.5*inch, 3*inch], hAlign='CENTER')
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ]))
        
        story.append(metadata_table)
        
        # Footer disclaimer
        story.append(Spacer(1, 50))
        disclaimer = """
        This report contains statistical analysis results based on survey data processing 
        using standardized methodological procedures. All estimates include appropriate 
        measures of uncertainty and follow official statistics guidelines.
        """
        story.append(Paragraph(disclaimer, self.styles['Footer']))
        
        return story
    
    def create_executive_summary(self, weighted_results: Dict[str, Any], config: Dict[str, Any]) -> List:
        """Create executive summary section."""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Key findings introduction
        intro_text = """
        This report presents comprehensive statistical analysis of survey data following 
        rigorous methodological standards. The analysis includes data quality assessment, 
        cleaning procedures, weighted statistical computations, and uncertainty quantification 
        suitable for official statistical releases.
        """
        story.append(Paragraph(intro_text, self.styles['BodyText']))
        story.append(Spacer(1, 12))
        
        # Key statistics summary
        story.append(Paragraph("Key Statistical Results", self.styles['SubsectionHeader']))
        
        if 'summary_stats' in weighted_results and not weighted_results['summary_stats'].empty:
            stats_df = weighted_results['summary_stats']
            
            # Create summary of top variables
            key_findings = []
            for i, (var_name, row) in enumerate(stats_df.iterrows()):
                if i >= 5:  # Limit to top 5 variables
                    break
                
                if 'weighted_mean' in row and not pd.isna(row['weighted_mean']):
                    mean_val = row['weighted_mean']
                    se_val = row.get('std_error', np.nan)
                    
                    if not pd.isna(se_val):
                        key_findings.append(f"• {var_name}: {mean_val:.3f} ± {se_val:.3f}")
                    else:
                        key_findings.append(f"• {var_name}: {mean_val:.3f}")
            
            if key_findings:
                findings_text = "\n".join(key_findings)
                story.append(Paragraph(findings_text, self.styles['BulletText']))
        
        # Data quality summary
        story.append(Spacer(1, 15))
        story.append(Paragraph("Data Quality Assessment", self.styles['SubsectionHeader']))
        
        quality_text = """
        Comprehensive data quality procedures were implemented including missing value 
        analysis, outlier detection, and validation rule application. All processing 
        steps are documented to ensure reproducibility and methodological transparency.
        """
        story.append(Paragraph(quality_text, self.styles['BodyText']))
        
        # Methodology note
        if config.get('include_methodology', True):
            story.append(Spacer(1, 15))
            methodology_note = """
            Statistical estimates incorporate survey design characteristics including 
            stratification and weighting procedures. Margins of error and confidence 
            intervals reflect the complex survey design and are appropriate for 
            statistical inference.
            """
            story.append(Paragraph(methodology_note, self.styles['MethodologyBox']))
        
        return story
    
    def create_data_overview_section(self, cleaned_data: pd.DataFrame, metadata: Dict[str, Any]) -> List:
        """Create detailed data overview section."""
        story = []
        
        story.append(Paragraph("Data Overview and Characteristics", self.styles['SectionHeader']))
        
        # Dataset summary
        data_summary = metadata.get('data_summary', {})
        
        overview_text = f"""
        The analytical dataset contains {data_summary.get('total_observations', 'N/A'):,} 
        observations across {data_summary.get('total_variables', 'N/A')} variables. 
        The overall data completeness rate is {data_summary.get('completeness_rate', 0):.1%}, 
        indicating {'high' if data_summary.get('completeness_rate', 0) > 0.9 else 'moderate' if data_summary.get('completeness_rate', 0) > 0.7 else 'low'} 
        data quality. The dataset includes {data_summary.get('numeric_variables', 'N/A')} 
        numeric variables and {data_summary.get('categorical_variables', 'N/A')} 
        categorical variables.
        """
        story.append(Paragraph(overview_text, self.styles['BodyText']))
        story.append(Spacer(1, 15))
        
        # Variable classification table
        story.append(Paragraph("Variable Classification", self.styles['SubsectionHeader']))
        
        # Create variable summary
        var_summary = self._create_variable_summary_table(cleaned_data)
        if var_summary:
            story.append(var_summary)
        
        story.append(Spacer(1, 15))
        
        # Missing data analysis
        story.append(Paragraph("Missing Data Analysis", self.styles['SubsectionHeader']))
        
        missing_analysis = self._create_missing_data_analysis(cleaned_data)
        story.extend(missing_analysis)
        
        return story
    
    def create_methodology_section(self, processing_log: List[Dict], config: Dict[str, Any]) -> List:
        """Create detailed methodology section."""
        story = []
        
        story.append(Paragraph("Methodology and Processing Procedures", self.styles['SectionHeader']))
        
        # Methodology overview
        methodology_intro = """
        This analysis follows established survey methodology standards and best practices 
        for official statistics. All procedures are designed to ensure data quality, 
        methodological consistency, and appropriate uncertainty quantification.
        """
        story.append(Paragraph(methodology_intro, self.styles['BodyText']))
        story.append(Spacer(1, 12))
        
        # Data processing workflow
        story.append(Paragraph("Data Processing Workflow", self.styles['SubsectionHeader']))
        
        if processing_log:
            workflow_steps = []
            for i, step in enumerate(processing_log, 1):
                step_name = step.get('step', 'Unknown Step')
                timestamp = step.get('timestamp', 'Unknown Time')
                
                if hasattr(timestamp, 'strftime'):
                    time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    time_str = str(timestamp)
                
                workflow_steps.append(f"{i}. {step_name} - {time_str}")
            
            workflow_text = "\n".join(workflow_steps)
            story.append(Paragraph(workflow_text, self.styles['BulletText']))
        
        story.append(Spacer(1, 15))
        
        # Statistical methods
        story.append(Paragraph("Statistical Methods", self.styles['SubsectionHeader']))
        
        methods_text = """
        <b>Weighting Procedures:</b> Survey weights are applied to account for differential 
        selection probabilities and non-response adjustments. Weights are normalized and 
        trimmed when necessary to control variance inflation.<br/><br/>
        
        <b>Variance Estimation:</b> Standard errors and confidence intervals account for 
        the complex survey design using appropriate variance estimation techniques. 
        Design effects are calculated and reported where relevant.<br/><br/>
        
        <b>Data Quality Control:</b> Comprehensive quality assurance procedures include 
        outlier detection, consistency checking, and validation rule application. 
        All quality control decisions are documented and reproducible.
        """
        story.append(Paragraph(methods_text, self.styles['BodyText']))
        
        return story
    
    def create_results_section(self, weighted_results: Dict[str, Any], config: Dict[str, Any]) -> List:
        """Create comprehensive results section."""
        story = []
        
        story.append(Paragraph("Statistical Results and Analysis", self.styles['SectionHeader']))
        
        # Summary statistics
        if 'summary_stats' in weighted_results and not weighted_results['summary_stats'].empty:
            story.append(Paragraph("Summary Statistics", self.styles['SubsectionHeader']))
            
            # Create formatted results table
            results_table = self._create_formatted_results_table(weighted_results['summary_stats'])
            if results_table:
                story.append(results_table)
                story.append(Spacer(1, 15))
        
        # Margins of error
        if 'margins_of_error' in weighted_results and not weighted_results['margins_of_error'].empty:
            story.append(Paragraph("Statistical Precision", self.styles['SubsectionHeader']))
            
            precision_text = """
            The following table presents margins of error for key estimates at the 95% 
            confidence level. These values represent the expected sampling variability 
            and should be considered when interpreting results.
            """
            story.append(Paragraph(precision_text, self.styles['BodyText']))
            story.append(Spacer(1, 8))
            
            moe_table = self._create_margin_of_error_table(weighted_results['margins_of_error'])
            if moe_table:
                story.append(moe_table)
                story.append(Spacer(1, 15))
        
        # Stratified results
        if 'stratified_results' in weighted_results and weighted_results['stratified_results']:
            story.append(Paragraph("Subgroup Analysis", self.styles['SubsectionHeader']))
            
            stratified_content = self._create_stratified_results_content(weighted_results['stratified_results'])
            story.extend(stratified_content)
        
        # Weight diagnostics
        if 'weight_diagnostics' in weighted_results:
            story.append(Paragraph("Survey Weight Diagnostics", self.styles['SubsectionHeader']))
            
            weight_content = self._create_weight_diagnostics_content(weighted_results['weight_diagnostics'])
            story.extend(weight_content)
        
        return story
    
    def create_data_quality_section(self, cleaned_data: pd.DataFrame, processing_log: List[Dict]) -> List:
        """Create data quality and diagnostics section."""
        story = []
        
        story.append(Paragraph("Data Quality Assessment and Diagnostics", self.styles['SectionHeader']))
        
        # Quality overview
        quality_intro = """
        Comprehensive data quality assessment procedures were implemented to ensure 
        the reliability and validity of statistical results. This section documents 
        all quality control measures and their outcomes.
        """
        story.append(Paragraph(quality_intro, self.styles['BodyText']))
        story.append(Spacer(1, 12))
        
        # Processing summary
        story.append(Paragraph("Data Processing Summary", self.styles['SubsectionHeader']))
        
        if processing_log:
            processing_summary = self._create_processing_summary(processing_log)
            story.extend(processing_summary)
        
        story.append(Spacer(1, 15))
        
        # Quality metrics
        story.append(Paragraph("Quality Metrics", self.styles['SubsectionHeader']))
        
        quality_metrics = self._create_quality_metrics_table(cleaned_data)
        if quality_metrics:
            story.append(quality_metrics)
        
        return story
    
    def _create_variable_summary_table(self, df: pd.DataFrame) -> Optional[Table]:
        """Create a summary table of variables."""
        if df.empty:
            return None
        
        # Analyze first 15 variables to keep table manageable
        columns_to_analyze = df.columns[:15]
        
        table_data = [['Variable', 'Type', 'Missing %', 'Unique Values']]
        
        for col in columns_to_analyze:
            var_type = self._classify_variable_type(df[col])
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            unique_count = df[col].nunique()
            
            table_data.append([
                col[:20] + ('...' if len(col) > 20 else ''),  # Truncate long names
                var_type,
                f"{missing_pct:.1f}%",
                f"{unique_count:,}"
            ])
        
        # Add note if more variables exist
        if len(df.columns) > 15:
            table_data.append(['...', '...', '...', f'({len(df.columns) - 15} more variables)'])
        
        table = Table(table_data, colWidths=[2*inch, 1*inch, 0.8*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (2, 0), (3, -1), 'RIGHT'),  # Right-align numeric columns
        ]))
        
        return table
    
    def _create_missing_data_analysis(self, df: pd.DataFrame) -> List:
        """Create missing data analysis content."""
        content = []
        
        missing_data = df.isnull().sum()
        missing_vars = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_vars) == 0:
            content.append(Paragraph("No missing data detected in the dataset.", self.styles['BodyText']))
            return content
        
        # Summary text
        total_missing = missing_data.sum()
        missing_pct = (total_missing / df.size) * 100
        
        summary_text = f"""
        A total of {total_missing:,} missing values were identified across 
        {len(missing_vars)} variables, representing {missing_pct:.2f}% of all data points. 
        The variables with the highest missing data rates are analyzed below.
        """
        content.append(Paragraph(summary_text, self.styles['BodyText']))
        content.append(Spacer(1, 10))
        
        # Missing data table (top 10 variables)
        if len(missing_vars) > 0:
            table_data = [['Variable', 'Missing Count', 'Missing %']]
            
            for var in missing_vars.head(10).index:
                missing_count = missing_vars[var]
                missing_rate = (missing_count / len(df)) * 100
                
                table_data.append([
                    var[:25] + ('...' if len(var) > 25 else ''),
                    f"{missing_count:,}",
                    f"{missing_rate:.1f}%"
                ])
            
            missing_table = Table(table_data, colWidths=[2.5*inch, 1*inch, 1*inch])
            missing_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (1, 0), (2, -1), 'RIGHT'),
            ]))
            
            content.append(missing_table)
        
        return content
    
    def _create_formatted_results_table(self, results_df: pd.DataFrame) -> Optional[Table]:
        """Create a professionally formatted results table."""
        if results_df.empty:
            return None
        
        # Limit to most important columns and first 10 variables
        important_cols = ['weighted_mean', 'std_error', 'ci_lower', 'ci_upper', 'count']
        available_cols = [col for col in important_cols if col in results_df.columns]
        
        if not available_cols:
            return None
        
        # Create table header
        header = ['Variable'] + [col.replace('_', ' ').title() for col in available_cols]
        table_data = [header]
        
        # Add data rows (limit to first 10 variables)
        for i, (var_name, row) in enumerate(results_df.iterrows()):
            if i >= 10:  # Limit table size
                break
            
            data_row = [var_name[:20] + ('...' if len(var_name) > 20 else '')]
            
            for col in available_cols:
                value = row[col]
                if pd.isna(value):
                    data_row.append('N/A')
                elif isinstance(value, (int, float)):
                    if col == 'count':
                        data_row.append(f"{int(value):,}")
                    else:
                        data_row.append(f"{value:.4f}")
                else:
                    data_row.append(str(value))
            
            table_data.append(data_row)
        
        # Create table
        num_cols = len(header)
        col_width = (self.page_width - 2 * self.margin) / num_cols
        col_widths = [col_width] * num_cols
        
        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),  # Right-align numeric columns
        ]))
        
        return table
    
    def _create_margin_of_error_table(self, moe_df: pd.DataFrame) -> Optional[Table]:
        """Create margin of error table."""
        if moe_df.empty:
            return None
        
        table_data = [['Variable', 'Margin of Error', 'Relative Error (%)']]
        
        # Add data for first 10 variables
        for i, (var_name, row) in enumerate(moe_df.iterrows()):
            if i >= 10:
                break
            
            moe_value = row['Margin of Error']
            if not pd.isna(moe_value):
                # Calculate relative error if possible (need estimate value)
                rel_error = "N/A"  # Would need the estimate to calculate this
                
                table_data.append([
                    var_name[:25] + ('...' if len(var_name) > 25 else ''),
                    f"{moe_value:.6f}",
                    rel_error
                ])
        
        if len(table_data) == 1:  # Only header
            return None
        
        table = Table(table_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 0), (2, -1), 'RIGHT'),
        ]))
        
        return table
    
    def _create_stratified_results_content(self, stratified_results: Dict[str, Any]) -> List:
        """Create content for stratified analysis results."""
        content = []
        
        strat_intro = """
        Subgroup analysis provides estimates for specific population segments. 
        Results are presented with appropriate confidence intervals reflecting 
        the reduced sample sizes within each stratum.
        """
        content.append(Paragraph(strat_intro, self.styles['BodyText']))
        content.append(Spacer(1, 10))
        
        # Process each stratification variable (limit to first 2)
        for i, (strat_var, strat_data) in enumerate(stratified_results.items()):
            if i >= 2:  # Limit to avoid overly long reports
                break
            
            content.append(Paragraph(f"Analysis by {strat_var}", self.styles['SubsectionHeader']))
            
            # Create simplified stratified results table
            strat_table = self._create_stratified_table(strat_var, strat_data)
            if strat_table:
                content.append(strat_table)
                content.append(Spacer(1, 15))
        
        return content
    
    def _create_stratified_table(self, strat_var: str, strat_data: Dict[str, Any]) -> Optional[Table]:
        """Create table for stratified results."""
        if not strat_data:
            return None
        
        # Extract first variable's results across strata
        first_var = list(strat_data.values())[0] if strat_data else {}
        if not first_var:
            return None
        
        analysis_var = list(first_var.keys())[0] if first_var else None
        if not analysis_var:
            return None
        
        table_data = [['Stratum', 'Estimate', 'Std Error', 'Count']]
        
        for stratum_name, stratum_results in strat_data.items():
            if analysis_var in stratum_results:
                var_results = stratum_results[analysis_var]
                
                if isinstance(var_results, dict):
                    estimate = var_results.get('weighted_mean', 'N/A')
                    std_err = var_results.get('std_error', 'N/A')
                    count = var_results.get('count', 'N/A')
                    
                    table_data.append([
                        str(stratum_name)[:15] + ('...' if len(str(stratum_name)) > 15 else ''),
                        f"{estimate:.4f}" if isinstance(estimate, (int, float)) else str(estimate),
                        f"{std_err:.4f}" if isinstance(std_err, (int, float)) else str(std_err),
                        f"{count:,}" if isinstance(count, (int, float)) else str(count)
                    ])
        
        if len(table_data) == 1:  # Only header
            return None
        
        table = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 0), (3, -1), 'RIGHT'),
        ]))
        
        return table
    
    def _create_weight_diagnostics_content(self, weight_diagnostics: Dict[str, Any]) -> List:
        """Create content for weight diagnostics."""
        content = []
        
        if not weight_diagnostics:
            content.append(Paragraph("No weight diagnostics available.", self.styles['BodyText']))
            return content
        
        diagnostics_text = f"""
        Survey weight analysis indicates a design effect of {weight_diagnostics.get('design_effect', 'N/A'):.3f} 
        and effective sample size of {weight_diagnostics.get('effective_sample_size', 'N/A'):.0f}. 
        The coefficient of variation for weights is {weight_diagnostics.get('cv_weight', 'N/A'):.3f}, 
        indicating {'moderate' if weight_diagnostics.get('cv_weight', 1) < 0.5 else 'high'} weight variability.
        """
        content.append(Paragraph(diagnostics_text, self.styles['BodyText']))
        
        # Weight statistics table
        weight_stats = [
            ['Statistic', 'Value'],
            ['Minimum Weight', f"{weight_diagnostics.get('min_weight', 'N/A'):.4f}"],
            ['Maximum Weight', f"{weight_diagnostics.get('max_weight', 'N/A'):.4f}"],
            ['Mean Weight', f"{weight_diagnostics.get('mean_weight', 'N/A'):.4f}"],
            ['Design Effect', f"{weight_diagnostics.get('design_effect', 'N/A'):.3f}"],
            ['Effective Sample Size', f"{weight_diagnostics.get('effective_sample_size', 'N/A'):.0f}"]
        ]
        
        weight_table = Table(weight_stats, colWidths=[2*inch, 1.5*inch])
        weight_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ]))
        
        content.append(Spacer(1, 10))
        content.append(weight_table)
        
        return content
    
    def _create_processing_summary(self, processing_log: List[Dict]) -> List:
        """Create processing summary content."""
        content = []
        
        processing_overview = f"""
        A total of {len(processing_log)} processing steps were completed. 
        The following summarizes key processing outcomes and quality control measures.
        """
        content.append(Paragraph(processing_overview, self.styles['BodyText']))
        content.append(Spacer(1, 10))
        
        # Extract key information from processing log
        for step in processing_log:
            step_name = step.get('step', 'Unknown')
            
            if 'report' in step and isinstance(step['report'], dict):
                report = step['report']
                
                step_summary = f"<b>{step_name}:</b> "
                
                if 'values_imputed' in report:
                    step_summary += f"Imputed {report['values_imputed']:,} missing values. "
                
                if 'outliers_detected' in report:
                    step_summary += f"Detected {report['outliers_detected']:,} outliers. "
                
                if 'rows_removed' in report:
                    step_summary += f"Removed {report['rows_removed']:,} rows. "
                
                content.append(Paragraph(step_summary, self.styles['BulletText']))
        
        return content
    
    def _create_quality_metrics_table(self, df: pd.DataFrame) -> Optional[Table]:
        """Create quality metrics table."""
        if df.empty:
            return None
        
        # Calculate quality metrics
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        duplicate_rows = df.duplicated().sum()
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Data Completeness', f"{completeness:.1%}"],
            ['Duplicate Rows', f"{duplicate_rows:,}"],
            ['Numeric Variables', f"{len(numeric_cols)}"],
            ['Total Data Points', f"{total_cells:,}"],
            ['Missing Data Points', f"{missing_cells:,}"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 1.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ]))
        
        return metrics_table
    
    def _classify_variable_type(self, series: pd.Series) -> str:
        """Classify variable type for display."""
        if pd.api.types.is_numeric_dtype(series):
            return 'Numeric'
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'Date/Time'
        elif series.nunique() <= 20:
            return 'Categorical'
        else:
            return 'Text'
    
    def create_footer_template(self, page_num: int, total_pages: int, report_date: str) -> str:
        """Create footer template for pages."""
        return f"Page {page_num} of {total_pages} | Generated: {report_date}"

class ExecutiveSummaryTemplate(StandardReportTemplate):
    """Template optimized for executive summary reports."""
    
    def __init__(self):
        super().__init__()
        self._setup_executive_styles()
    
    def _setup_executive_styles(self):
        """Setup styles optimized for executive summary."""
        
        # Larger, more prominent title
        self.styles.add(ParagraphStyle(
            name='ExecutiveTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=40,
            alignment=TA_CENTER,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        ))
        
        # Highlight boxes for key findings
        self.styles.add(ParagraphStyle(
            name='KeyFinding',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceBefore=15,
            spaceAfter=15,
            leftIndent=30,
            rightIndent=30,
            borderWidth=2,
            borderColor=colors.darkgreen,
            borderPadding=15,
            backColor=colors.lightgreen,
            fontName='Helvetica-Bold'
        ))

class TechnicalReportTemplate(StandardReportTemplate):
    """Template optimized for technical/detailed reports."""
    
    def __init__(self):
        super().__init__()
        self._setup_technical_styles()
    
    def _setup_technical_styles(self):
        """Setup styles for technical reports."""
        
        # Code/formula style
        self.styles.add(ParagraphStyle(
            name='CodeBlock',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceBefore=10,
            spaceAfter=10,
            leftIndent=20,
            fontName='Courier',
            backColor=colors.lightgrey,
            borderWidth=1,
            borderColor=colors.grey,
            borderPadding=10
        ))
        
        # Technical note style
        self.styles.add(ParagraphStyle(
            name='TechnicalNote',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceBefore=10,
            spaceAfter=10,
            leftIndent=15,
            rightIndent=15,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.blue,
            borderPadding=8,
            backColor=colors.aliceblue
        ))

def get_report_template(template_name: str) -> StandardReportTemplate:
    """Factory function to get appropriate report template."""
    
    templates = {
        'Standard': StandardReportTemplate,
        'Executive Summary': ExecutiveSummaryTemplate,
        'Technical Report': TechnicalReportTemplate
    }
    
    template_class = templates.get(template_name, StandardReportTemplate)
    return template_class()

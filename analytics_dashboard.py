"""
Analytics Dashboard - Comprehensive monitoring dashboard for agent performance, error rates, and pick quality
Real-time visualization of system health, performance metrics, and analytics insights
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
import json

from agent_logger import get_logger
from agent_monitor import get_monitor, PerformanceMetrics, PickAnalytics
from agent_analytics_tracker import get_analytics_tracker, QualityMetrics, PatternInsight
from agent_prompt_manager import get_prompt_manager
from ui_error_handler import get_ui_error_handler

class AnalyticsDashboard:
    """Comprehensive analytics dashboard for agent monitoring"""
    
    def __init__(self):
        self.logger = get_logger()
        self.monitor = get_monitor()
        self.analytics_tracker = get_analytics_tracker()
        self.prompt_manager = get_prompt_manager()
        self.ui_error_handler = get_ui_error_handler()
        
        # Dashboard configuration
        self.refresh_interval = 30  # seconds
        self.default_time_range = 7  # days
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'error': '#d62728',
            'info': '#17becf',
            'neutral': '#7f7f7f'
        }
    
    def render_dashboard(self):
        """Render the complete analytics dashboard"""
        
        st.title("🎯 BetFinder AI Analytics Dashboard")
        
        # Sidebar controls
        self._render_sidebar_controls()
        
        # Main dashboard content
        self._render_system_overview()
        
        # Tabs for different analytics views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Performance", "🎯 Quality", "🔍 Patterns", "📝 Prompts", "⚠️ Errors"
        ])
        
        with tab1:
            self._render_performance_tab()
        
        with tab2:
            self._render_quality_tab()
        
        with tab3:
            self._render_patterns_tab()
        
        with tab4:
            self._render_prompts_tab()
        
        with tab5:
            self._render_errors_tab()
        
        # Auto-refresh
        self._setup_auto_refresh()
    
    def _render_sidebar_controls(self):
        """Render sidebar controls for dashboard configuration"""
        
        st.sidebar.header("🎛️ Dashboard Controls")
        
        # Time range selector
        time_range = st.sidebar.selectbox(
            "Time Range",
            options=[1, 3, 7, 14, 30],
            index=2,  # Default to 7 days
            format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}"
        )
        st.session_state.dashboard_time_range = time_range
        
        # Agent filter
        all_agents = list(set([
            f"{metrics.agent_name}:{metrics.sport}"
            for metrics in self.monitor.performance_cache.values()
        ]))
        
        selected_agents = st.sidebar.multiselect(
            "Filter Agents",
            options=all_agents,
            default=all_agents[:5] if len(all_agents) > 5 else all_agents
        )
        st.session_state.dashboard_selected_agents = selected_agents
        
        # Metric selector
        selected_metrics = st.sidebar.multiselect(
            "Display Metrics",
            options=[
                "Win Rate", "Confidence", "Response Time", "Error Rate",
                "Profit/Loss", "Volume", "ROI", "Sharpe Ratio"
            ],
            default=["Win Rate", "Confidence", "Error Rate", "ROI"]
        )
        st.session_state.dashboard_selected_metrics = selected_metrics
        
        # Refresh controls
        st.sidebar.markdown("---")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Refresh Now"):
                st.experimental_rerun()
        
        with col2:
            auto_refresh = st.checkbox("Auto Refresh", value=True)
            st.session_state.dashboard_auto_refresh = auto_refresh
        
        # Export controls
        st.sidebar.markdown("---")
        st.sidebar.header("📤 Export Data")
        
        if st.sidebar.button("Download Analytics Report"):
            self._export_analytics_report()
        
        if st.sidebar.button("Download Performance Data"):
            self._export_performance_data()
        
        # System status
        self.ui_error_handler.display_system_status()
    
    def _render_system_overview(self):
        """Render system overview section"""
        
        st.header("🏠 System Overview")
        
        # Get system health
        health = self.monitor.get_system_health()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="System Status",
                value=health['status'].title(),
                delta=f"{health['uptime_percentage']:.1f}% uptime"
            )
        
        with col2:
            st.metric(
                label="Active Agents",
                value=f"{health['healthy_agents']}/{health['total_agents']}",
                delta=f"{health['error_agents']} with issues" if health['error_agents'] > 0 else "All healthy"
            )
        
        with col3:
            st.metric(
                label="Avg Response Time",
                value=f"{health['avg_response_time']:.0f}ms",
                delta="Good" if health['avg_response_time'] < 1000 else "Slow"
            )
        
        with col4:
            st.metric(
                label="Requests/Hour",
                value=health['total_requests_hour'],
                delta=f"{health['total_picks_hour']} picks"
            )
        
        # System health chart
        if health['status'] == 'critical':
            st.error("🚨 System is experiencing critical issues")
        elif health['status'] == 'warning':
            st.warning("⚠️ System has some issues but is functional")
        else:
            st.success("✅ System is operating normally")
    
    def _render_performance_tab(self):
        """Render performance analytics tab"""
        
        st.header("📊 Performance Analytics")
        
        time_range = st.session_state.get('dashboard_time_range', 7)
        selected_agents = st.session_state.get('dashboard_selected_agents', [])
        
        if not selected_agents:
            st.info("Please select agents in the sidebar to view performance data")
            return
        
        # Get performance data
        performance_data = []
        for agent_key in selected_agents:
            agent_name, sport = agent_key.split(':', 1)
            metrics = self.monitor.get_agent_performance(agent_name, sport)
            if agent_key in metrics:
                performance_data.append({
                    'agent': agent_key,
                    'agent_name': agent_name,
                    'sport': sport,
                    **metrics[agent_key].__dict__
                })
        
        if not performance_data:
            st.info("No performance data available for selected agents")
            return
        
        df = pd.DataFrame(performance_data)
        
        # Performance overview table
        st.subheader("📋 Performance Summary")
        
        summary_df = df[[
            'agent', 'win_rate', 'avg_confidence', 'error_rate', 
            'total_profit', 'roi', 'picks_generated'
        ]].copy()
        
        # Format for display
        summary_df['win_rate'] = summary_df['win_rate'].apply(lambda x: f"{x:.1%}")
        summary_df['avg_confidence'] = summary_df['avg_confidence'].apply(lambda x: f"{x:.2f}")
        summary_df['error_rate'] = summary_df['error_rate'].apply(lambda x: f"{x:.1%}")
        summary_df['total_profit'] = summary_df['total_profit'].apply(lambda x: f"${x:.2f}")
        summary_df['roi'] = summary_df['roi'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(summary_df, use_container_width=True)
        
        # Performance charts
        self._render_performance_charts(df)
        
        # Detailed metrics
        self._render_detailed_metrics(df)
    
    def _render_performance_charts(self, df: pd.DataFrame):
        """Render performance visualization charts"""
        
        st.subheader("📈 Performance Visualization")
        
        # Win rate comparison
        fig_win_rate = px.bar(
            df, x='agent', y='win_rate',
            title="Win Rate by Agent",
            color='win_rate',
            color_continuous_scale='RdYlGn'
        )
        fig_win_rate.update_layout(
            xaxis_title="Agent",
            yaxis_title="Win Rate",
            yaxis_tickformat='.1%'
        )
        st.plotly_chart(fig_win_rate, use_container_width=True)
        
        # Multi-metric comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence vs Win Rate scatter
            fig_scatter = px.scatter(
                df, x='avg_confidence', y='win_rate',
                size='picks_generated',
                color='sport',
                hover_data=['agent'],
                title="Confidence vs Win Rate"
            )
            fig_scatter.update_layout(
                xaxis_title="Average Confidence",
                yaxis_title="Win Rate",
                yaxis_tickformat='.1%'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # ROI comparison
            fig_roi = px.bar(
                df, x='agent', y='roi',
                title="Return on Investment (ROI)",
                color='roi',
                color_continuous_scale='RdYlGn'
            )
            fig_roi.update_layout(
                xaxis_title="Agent",
                yaxis_title="ROI",
                yaxis_tickformat='.1%'
            )
            st.plotly_chart(fig_roi, use_container_width=True)
        
        # Error rate analysis
        error_data = df[df['error_rate'] > 0]
        if not error_data.empty:
            fig_errors = px.bar(
                error_data, x='agent', y='error_rate',
                title="Error Rates by Agent",
                color='error_rate',
                color_continuous_scale='Reds'
            )
            fig_errors.update_layout(
                xaxis_title="Agent",
                yaxis_title="Error Rate",
                yaxis_tickformat='.1%'
            )
            st.plotly_chart(fig_errors, use_container_width=True)
    
    def _render_detailed_metrics(self, df: pd.DataFrame):
        """Render detailed performance metrics"""
        
        st.subheader("🔍 Detailed Metrics")
        
        # Select agent for detailed view
        selected_agent = st.selectbox(
            "Select Agent for Detailed Analysis",
            options=df['agent'].tolist()
        )
        
        if selected_agent:
            agent_data = df[df['agent'] == selected_agent].iloc[0]
            agent_name, sport = selected_agent.split(':', 1)
            
            # Detailed metrics display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Requests", f"{agent_data['total_requests']:,}")
                st.metric("Successful Requests", f"{agent_data['successful_requests']:,}")
                st.metric("Avg Response Time", f"{agent_data['avg_response_time']:.0f}ms")
            
            with col2:
                st.metric("Picks Generated", f"{agent_data['picks_generated']:,}")
                st.metric("Picks Resolved", f"{agent_data['picks_resolved']:,}")
                st.metric("Win Rate", f"{agent_data['win_rate']:.1%}")
            
            with col3:
                st.metric("Total Profit", f"${agent_data['total_profit']:.2f}")
                st.metric("ROI", f"{agent_data['roi']:.1%}")
                st.metric("Sharpe Ratio", f"{agent_data['sharpe_ratio']:.2f}")
            
            # Get recent picks for this agent
            recent_picks = self.monitor.get_pick_analytics(
                agent_name=agent_name,
                sport=sport,
                days=st.session_state.get('dashboard_time_range', 7)
            )
            
            if recent_picks:
                self._render_pick_analysis(recent_picks)
    
    def _render_pick_analysis(self, picks: List[PickAnalytics]):
        """Render pick-level analysis"""
        
        st.subheader("🎯 Recent Picks Analysis")
        
        if not picks:
            st.info("No recent picks found")
            return
        
        # Convert to DataFrame
        picks_data = []
        for pick in picks:
            picks_data.append({
                'pick_id': pick.pick_id[:8],  # Shortened ID
                'player': pick.player_name,
                'stat': pick.stat_category,
                'line': pick.line_value,
                'prediction': pick.prediction,
                'confidence': pick.confidence,
                'outcome': pick.outcome or 'Pending',
                'profit_loss': pick.profit_loss or 0,
                'generated_at': pick.generated_at
            })
        
        picks_df = pd.DataFrame(picks_data)
        
        # Recent picks table
        st.dataframe(
            picks_df.sort_values('generated_at', ascending=False).head(10),
            use_container_width=True
        )
        
        # Picks performance chart
        resolved_picks = picks_df[picks_df['outcome'].isin(['win', 'loss'])]
        
        if not resolved_picks.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Outcome distribution
                outcome_counts = resolved_picks['outcome'].value_counts()
                fig_outcomes = px.pie(
                    values=outcome_counts.values,
                    names=outcome_counts.index,
                    title="Pick Outcomes"
                )
                st.plotly_chart(fig_outcomes, use_container_width=True)
            
            with col2:
                # Confidence distribution
                fig_conf = px.histogram(
                    resolved_picks, x='confidence',
                    title="Confidence Distribution",
                    nbins=10
                )
                fig_conf.update_layout(
                    xaxis_title="Confidence",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_conf, use_container_width=True)
    
    def _render_quality_tab(self):
        """Render quality analytics tab"""
        
        st.header("🎯 Quality Analytics")
        
        selected_agents = st.session_state.get('dashboard_selected_agents', [])
        
        if not selected_agents:
            st.info("Please select agents in the sidebar to view quality data")
            return
        
        # Get quality metrics
        quality_data = []
        for agent_key in selected_agents:
            agent_name, sport = agent_key.split(':', 1)
            
            # Analyze quality if not already done
            quality_metrics = self.analytics_tracker.analyze_agent_quality(agent_name, sport)
            
            if quality_metrics:
                quality_data.append({
                    'agent': agent_key,
                    'agent_name': agent_name,
                    'sport': sport,
                    **quality_metrics.__dict__
                })
        
        if not quality_data:
            st.info("No quality data available")
            return
        
        df = pd.DataFrame(quality_data)
        
        # Quality overview
        st.subheader("📊 Quality Overview")
        
        # Quality metrics table
        quality_summary = df[[
            'agent', 'overall_accuracy', 'confidence_calibration',
            'prediction_bias', 'performance_volatility', 'value_creation_rate'
        ]].copy()
        
        # Format for display
        quality_summary['overall_accuracy'] = quality_summary['overall_accuracy'].apply(lambda x: f"{x:.1%}")
        quality_summary['confidence_calibration'] = quality_summary['confidence_calibration'].apply(lambda x: f"{x:.2f}")
        quality_summary['prediction_bias'] = quality_summary['prediction_bias'].apply(lambda x: f"{x:.2f}")
        quality_summary['performance_volatility'] = quality_summary['performance_volatility'].apply(lambda x: f"{x:.2f}")
        quality_summary['value_creation_rate'] = quality_summary['value_creation_rate'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(quality_summary, use_container_width=True)
        
        # Quality visualization
        self._render_quality_charts(df)
    
    def _render_quality_charts(self, df: pd.DataFrame):
        """Render quality visualization charts"""
        
        st.subheader("📈 Quality Visualization")
        
        # Quality radar chart
        if len(df) > 0:
            # Select agent for radar chart
            selected_agent = st.selectbox(
                "Select Agent for Quality Radar",
                options=df['agent'].tolist(),
                key="quality_radar_agent"
            )
            
            if selected_agent:
                agent_data = df[df['agent'] == selected_agent].iloc[0]
                
                # Create radar chart
                categories = [
                    'Overall Accuracy', 'Confidence Calibration', 'Value Creation',
                    'Consistency', 'Improvement Trend', 'Adaptation Speed'
                ]
                
                values = [
                    agent_data['overall_accuracy'],
                    agent_data['confidence_calibration'],
                    agent_data['value_creation_rate'],
                    1.0 - agent_data['performance_volatility'],  # Convert volatility to consistency
                    max(0, agent_data['improvement_trend'] + 0.5),  # Normalize improvement trend
                    agent_data['adaptation_speed']
                ]
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=selected_agent
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title=f"Quality Profile: {selected_agent}"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
        
        # Quality comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy vs Calibration
            fig_acc_cal = px.scatter(
                df, x='confidence_calibration', y='overall_accuracy',
                size='value_creation_rate',
                color='sport',
                hover_data=['agent'],
                title="Accuracy vs Calibration"
            )
            st.plotly_chart(fig_acc_cal, use_container_width=True)
        
        with col2:
            # Bias analysis
            fig_bias = px.bar(
                df, x='agent', y='prediction_bias',
                title="Prediction Bias Analysis",
                color='prediction_bias',
                color_continuous_scale='RdBu'
            )
            fig_bias.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_bias, use_container_width=True)
    
    def _render_patterns_tab(self):
        """Render pattern analysis tab"""
        
        st.header("🔍 Pattern Analysis")
        
        # Run pattern detection
        if st.button("🔍 Detect New Patterns"):
            with st.spinner("Analyzing patterns..."):
                new_insights = self.analytics_tracker.detect_patterns()
                if new_insights:
                    st.success(f"Found {len(new_insights)} new patterns!")
                else:
                    st.info("No new patterns detected")
        
        # Display existing patterns
        recent_patterns = self.analytics_tracker.pattern_insights[-20:]  # Last 20 patterns
        
        if not recent_patterns:
            st.info("No patterns detected yet. Run pattern detection to analyze agent behavior.")
            return
        
        # Pattern summary
        st.subheader("📋 Recent Pattern Insights")
        
        for i, pattern in enumerate(reversed(recent_patterns)):
            with st.expander(f"{pattern.pattern_type.replace('_', ' ').title()}: {pattern.description}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Confidence:** {pattern.confidence:.1%}")
                    st.write(f"**Impact Score:** {pattern.impact_score:.2f}")
                    st.write(f"**Affected Picks:** {len(pattern.affected_picks)}")
                
                with col2:
                    st.write(f"**Recommendation:** {pattern.recommendation}")
                    
                    if pattern.impact_score < -0.2:
                        st.error("⚠️ High negative impact - requires attention")
                    elif pattern.impact_score > 0.2:
                        st.success("✅ Positive pattern - leverage this behavior")
                    else:
                        st.info("ℹ️ Neutral impact - monitor for changes")
        
        # Pattern analysis charts
        self._render_pattern_charts(recent_patterns)
    
    def _render_pattern_charts(self, patterns: List[PatternInsight]):
        """Render pattern analysis charts"""
        
        if not patterns:
            return
        
        st.subheader("📊 Pattern Analysis")
        
        # Pattern type distribution
        pattern_types = {}
        for pattern in patterns:
            pattern_type = pattern.pattern_type.replace('_', ' ').title()
            if pattern_type not in pattern_types:
                pattern_types[pattern_type] = 0
            pattern_types[pattern_type] += 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_types = px.pie(
                values=list(pattern_types.values()),
                names=list(pattern_types.keys()),
                title="Pattern Type Distribution"
            )
            st.plotly_chart(fig_types, use_container_width=True)
        
        with col2:
            # Impact score distribution
            impact_scores = [p.impact_score for p in patterns]
            fig_impact = px.histogram(
                x=impact_scores,
                nbins=10,
                title="Pattern Impact Score Distribution"
            )
            fig_impact.add_vline(x=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_impact, use_container_width=True)
    
    def _render_prompts_tab(self):
        """Render prompt management tab"""
        
        st.header("📝 Prompt Management")
        
        # Get prompt performance report
        prompt_report = self.prompt_manager.get_performance_report(
            days=st.session_state.get('dashboard_time_range', 7)
        )
        
        # Prompt summary
        st.subheader("📊 Prompt Performance Summary")
        
        summary = prompt_report['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Templates", summary['total_templates'])
        
        with col2:
            st.metric("Active Templates", summary['active_templates'])
        
        with col3:
            st.metric("Total Usage", summary['total_usage'])
        
        with col4:
            st.metric("Avg Success Rate", f"{summary['avg_success_rate']:.1%}")
        
        # Active A/B tests
        if prompt_report['ab_tests']:
            st.subheader("🧪 Active A/B Tests")
            
            for test_id, test_data in prompt_report['ab_tests'].items():
                with st.expander(f"Test: {test_data['name']}"):
                    st.write(f"**Status:** {test_data['status']}")
                    st.write(f"**Statistical Significance:** {test_data['statistical_significance']}")
                    
                    if test_data['results']:
                        results_df = pd.DataFrame(test_data['results']).T
                        st.dataframe(results_df)
        
        # Template performance
        if prompt_report['templates']:
            st.subheader("📋 Template Performance")
            
            templates_data = []
            for template_id, template_data in prompt_report['templates'].items():
                templates_data.append({
                    'template_id': template_id[:8],
                    'name': template_data['name'],
                    'version': template_data['version'],
                    'sport': template_data['sport'],
                    'is_active': template_data['is_active'],
                    'usage': template_data['performance']['total_uses'],
                    'win_rate': template_data['performance']['win_rate']
                })
            
            if templates_data:
                templates_df = pd.DataFrame(templates_data)
                st.dataframe(templates_df, use_container_width=True)
        
        # Prompt optimization recommendations
        st.subheader("💡 Optimization Recommendations")
        
        recommendations = self.prompt_manager.optimize_prompts()
        
        if recommendations:
            for rec in recommendations:
                st.write(f"• {rec}")
        else:
            st.info("No optimization recommendations at this time")
    
    def _render_errors_tab(self):
        """Render error analysis tab"""
        
        st.header("⚠️ Error Analysis")
        
        # Get error summary
        error_summary = self.logger.get_error_summary(hours=24)
        
        # Error overview
        st.subheader("📊 Error Overview (Last 24 Hours)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Errors", error_summary['total_errors'])
        
        with col2:
            st.metric("Error Rate", f"{error_summary['error_rate']:.1%}")
        
        with col3:
            st.metric("Affected Agents", len(error_summary['affected_agents']))
        
        # Error type distribution
        if error_summary['error_types']:
            st.subheader("🔍 Error Type Distribution")
            
            error_types_df = pd.DataFrame([
                {'error_type': k, 'count': v}
                for k, v in error_summary['error_types'].items()
            ])
            
            fig_error_types = px.bar(
                error_types_df, x='error_type', y='count',
                title="Error Types",
                color='count',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_error_types, use_container_width=True)
        
        # Recent errors
        recent_logs = self.logger.get_recent_logs(
            limit=50,
            event_type=None  # All events
        )
        
        error_logs = [log for log in recent_logs if log.level.value == 'ERROR']
        
        if error_logs:
            st.subheader("📝 Recent Errors")
            
            for log in error_logs[-10:]:  # Last 10 errors
                with st.expander(f"{log.error_type or 'Unknown'}: {log.agent_name}:{log.sport}"):
                    st.write(f"**Time:** {log.timestamp}")
                    st.write(f"**Agent:** {log.agent_name}")
                    st.write(f"**Sport:** {log.sport}")
                    st.write(f"**Error:** {log.error_message}")
                    
                    if log.error_traceback:
                        st.code(log.error_traceback)
        
        # Error trends
        self._render_error_trends()
    
    def _render_error_trends(self):
        """Render error trend analysis"""
        
        st.subheader("📈 Error Trends")
        
        # Get recent logs for trend analysis
        recent_logs = self.logger.get_recent_logs(limit=1000)
        
        if not recent_logs:
            st.info("No log data available for trend analysis")
            return
        
        # Convert to DataFrame
        logs_data = []
        for log in recent_logs:
            logs_data.append({
                'timestamp': log.timestamp,
                'level': log.level.value,
                'agent': f"{log.agent_name}:{log.sport}",
                'error_type': log.error_type or 'Unknown',
                'is_error': log.level.value in ['ERROR', 'CRITICAL']
            })
        
        logs_df = pd.DataFrame(logs_data)
        logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
        
        # Error rate over time
        error_logs = logs_df[logs_df['is_error']]
        
        if not error_logs.empty:
            # Group by hour
            error_logs['hour'] = error_logs['timestamp'].dt.floor('H')
            hourly_errors = error_logs.groupby('hour').size().reset_index(name='error_count')
            
            fig_trends = px.line(
                hourly_errors, x='hour', y='error_count',
                title="Error Rate Trend (Hourly)",
                markers=True
            )
            st.plotly_chart(fig_trends, use_container_width=True)
    
    def _setup_auto_refresh(self):
        """Setup auto-refresh functionality"""
        
        if st.session_state.get('dashboard_auto_refresh', True):
            # Note: Streamlit doesn't have native auto-refresh
            # This would need to be implemented with periodic rerun
            pass
    
    def _export_analytics_report(self):
        """Export comprehensive analytics report"""
        
        try:
            # Generate comprehensive report
            report_file = f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Export from analytics tracker
            output_path = self.analytics_tracker.export_analytics_report(
                report_file,
                agent_name=None,
                sport=None
            )
            
            st.success(f"Analytics report exported to: {output_path}")
            
            # Provide download link (if running in a server environment)
            with open(output_path, 'r') as f:
                st.download_button(
                    label="Download Analytics Report",
                    data=f.read(),
                    file_name=report_file,
                    mime="application/json"
                )
        
        except Exception as e:
            st.error(f"Failed to export analytics report: {str(e)}")
    
    def _export_performance_data(self):
        """Export performance data"""
        
        try:
            # Export performance data
            export_file = f"performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            output_path = self.monitor.export_analytics(export_file, format='json')
            
            st.success(f"Performance data exported to: {output_path}")
            
            # Provide download link
            with open(output_path, 'r') as f:
                st.download_button(
                    label="Download Performance Data",
                    data=f.read(),
                    file_name=export_file,
                    mime="application/json"
                )
        
        except Exception as e:
            st.error(f"Failed to export performance data: {str(e)}")

# Global dashboard instance
_global_dashboard: Optional[AnalyticsDashboard] = None

def get_dashboard() -> AnalyticsDashboard:
    """Get the global dashboard instance"""
    global _global_dashboard
    if _global_dashboard is None:
        _global_dashboard = AnalyticsDashboard()
    return _global_dashboard

def render_analytics_dashboard():
    """Render the analytics dashboard"""
    dashboard = get_dashboard()
    dashboard.render_dashboard()
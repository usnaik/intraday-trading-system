"""
Fundamental Analysis Module for the Intraday Trading System
Analyzes company fundamentals and generates fundamental-based trading scores
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from loguru import logger

from ..models.data_models import FundamentalMetrics, SignalType
from ..data.yahoo_data_fetcher import YahooDataFetcher


class FundamentalAnalyzer:
    """Fundamental analysis engine for trading signals"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.fundamental_config = self.config.get('fundamental_metrics', {})
        self.data_fetcher = YahooDataFetcher()
        
        # Scoring weights for different fundamental categories
        self.category_weights = {
            'profitability': 0.30,
            'valuation': 0.25,
            'growth': 0.25,
            'financial_health': 0.20
        }
        
        # Industry benchmarks (these would ideally come from a database)
        self.industry_benchmarks = self._initialize_industry_benchmarks()
        
    def _initialize_industry_benchmarks(self) -> Dict:
        """Initialize industry benchmark values for comparison"""
        return {
            'technology': {
                'pe_ratio': {'good': 25, 'average': 35, 'poor': 50},
                'pb_ratio': {'good': 3, 'average': 5, 'poor': 8},
                'roe': {'good': 20, 'average': 15, 'poor': 10},
                'debt_to_equity': {'good': 0.3, 'average': 0.6, 'poor': 1.0},
                'revenue_growth': {'good': 15, 'average': 10, 'poor': 5}
            },
            'finance': {
                'pe_ratio': {'good': 12, 'average': 15, 'poor': 20},
                'pb_ratio': {'good': 1.0, 'average': 1.5, 'poor': 2.0},
                'roe': {'good': 15, 'average': 12, 'poor': 8},
                'debt_to_equity': {'good': 0.8, 'average': 1.2, 'poor': 1.8},
                'revenue_growth': {'good': 8, 'average': 5, 'poor': 2}
            },
            'healthcare': {
                'pe_ratio': {'good': 20, 'average': 28, 'poor': 40},
                'pb_ratio': {'good': 2, 'average': 4, 'poor': 6},
                'roe': {'good': 18, 'average': 14, 'poor': 10},
                'debt_to_equity': {'good': 0.4, 'average': 0.7, 'poor': 1.2},
                'revenue_growth': {'good': 12, 'average': 8, 'poor': 4}
            },
            'energy': {
                'pe_ratio': {'good': 15, 'average': 20, 'poor': 30},
                'pb_ratio': {'good': 1.5, 'average': 2.5, 'poor': 3.5},
                'roe': {'good': 12, 'average': 8, 'poor': 4},
                'debt_to_equity': {'good': 0.5, 'average': 1.0, 'poor': 1.5},
                'revenue_growth': {'good': 10, 'average': 5, 'poor': 0}
            },
            'default': {  # Used when industry is not specified
                'pe_ratio': {'good': 18, 'average': 25, 'poor': 35},
                'pb_ratio': {'good': 2, 'average': 3, 'poor': 5},
                'roe': {'good': 15, 'average': 12, 'poor': 8},
                'debt_to_equity': {'good': 0.5, 'average': 1.0, 'poor': 1.5},
                'revenue_growth': {'good': 10, 'average': 6, 'poor': 2}
            }
        }
    
    async def analyze_fundamentals(self, symbols: List[str]) -> Dict[str, FundamentalMetrics]:
        """
        Analyze fundamental metrics for multiple symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with fundamental metrics for each symbol
        """
        logger.info(f"Analyzing fundamentals for symbols: {symbols}")
        
        # Fetch fundamental data
        fundamental_data = self.data_fetcher.get_fundamental_data(symbols)
        
        results = {}
        
        for symbol in symbols:
            try:
                if symbol in fundamental_data:
                    raw_data = fundamental_data[symbol]
                    
                    # Create FundamentalMetrics object
                    metrics = FundamentalMetrics(
                        symbol=symbol,
                        timestamp=raw_data.get('timestamp', datetime.now()),
                        
                        # Profitability metrics
                        roe=raw_data.get('roe'),
                        roa=raw_data.get('roa'),
                        profit_margin=raw_data.get('profit_margin'),
                        operating_margin=raw_data.get('operating_margin'),
                        
                        # Valuation metrics
                        pe_ratio=raw_data.get('pe_ratio'),
                        pb_ratio=raw_data.get('pb_ratio'),
                        ev_ebitda=raw_data.get('ev_ebitda'),
                        
                        # Growth metrics
                        revenue_growth=raw_data.get('revenue_growth'),
                        earnings_growth=raw_data.get('earnings_growth'),
                        
                        # Financial health metrics
                        debt_to_equity=raw_data.get('debt_to_equity'),
                        current_ratio=raw_data.get('current_ratio'),
                        quick_ratio=raw_data.get('quick_ratio'),
                        
                        # Market data
                        market_cap=raw_data.get('market_cap'),
                        beta=raw_data.get('beta')
                    )
                    
                    results[symbol] = metrics
                    logger.debug(f"Fundamental analysis completed for {symbol}")
                    
                else:
                    logger.warning(f"No fundamental data available for {symbol}")
                    # Create empty metrics object
                    results[symbol] = FundamentalMetrics(
                        symbol=symbol,
                        timestamp=datetime.now()
                    )
                    
            except Exception as e:
                logger.error(f"Error analyzing fundamentals for {symbol}: {str(e)}")
                results[symbol] = FundamentalMetrics(
                    symbol=symbol,
                    timestamp=datetime.now()
                )
        
        logger.info(f"Fundamental analysis completed for {len(results)} symbols")
        return results
    
    def generate_fundamental_score(self, metrics: FundamentalMetrics, industry: str = None) -> float:
        """
        Generate a fundamental score (0-100) based on company metrics
        
        Args:
            metrics: Fundamental metrics for the company
            industry: Industry classification for benchmarking
            
        Returns:
            Fundamental score from 0-100
        """
        try:
            logger.debug(f"Generating fundamental score for {metrics.symbol}")
            
            # Get industry benchmarks
            benchmarks = self.industry_benchmarks.get(industry, self.industry_benchmarks['default'])
            
            # Calculate category scores
            profitability_score = self._calculate_profitability_score(metrics, benchmarks)
            valuation_score = self._calculate_valuation_score(metrics, benchmarks)
            growth_score = self._calculate_growth_score(metrics, benchmarks)
            financial_health_score = self._calculate_financial_health_score(metrics, benchmarks)
            
            # Calculate weighted composite score
            composite_score = (
                profitability_score * self.category_weights['profitability'] +
                valuation_score * self.category_weights['valuation'] +
                growth_score * self.category_weights['growth'] +
                financial_health_score * self.category_weights['financial_health']
            )
            
            logger.debug(f"Fundamental score for {metrics.symbol}: {composite_score:.2f}")
            return max(0.0, min(100.0, composite_score))
            
        except Exception as e:
            logger.error(f"Error generating fundamental score for {metrics.symbol}: {str(e)}")
            return 50.0  # Return neutral score on error
    
    def _calculate_profitability_score(self, metrics: FundamentalMetrics, benchmarks: Dict) -> float:
        """Calculate profitability-based score"""
        try:
            score = 0
            valid_metrics = 0
            
            # ROE (Return on Equity) - 30 points
            if metrics.roe is not None and not np.isnan(metrics.roe):
                roe_pct = metrics.roe * 100 if metrics.roe < 1 else metrics.roe
                if roe_pct >= benchmarks['roe']['good']:
                    score += 30
                elif roe_pct >= benchmarks['roe']['average']:
                    score += 20
                elif roe_pct >= benchmarks['roe']['poor']:
                    score += 10
                valid_metrics += 1
            
            # ROA (Return on Assets) - 25 points
            if metrics.roa is not None and not np.isnan(metrics.roa):
                roa_pct = metrics.roa * 100 if metrics.roa < 1 else metrics.roa
                if roa_pct >= 8:  # Good ROA
                    score += 25
                elif roa_pct >= 5:  # Average ROA
                    score += 15
                elif roa_pct >= 2:  # Poor but positive ROA
                    score += 8
                valid_metrics += 1
            
            # Profit Margin - 25 points
            if metrics.profit_margin is not None and not np.isnan(metrics.profit_margin):
                margin_pct = metrics.profit_margin * 100 if metrics.profit_margin < 1 else metrics.profit_margin
                if margin_pct >= 15:  # Excellent margin
                    score += 25
                elif margin_pct >= 10:  # Good margin
                    score += 20
                elif margin_pct >= 5:  # Average margin
                    score += 12
                elif margin_pct > 0:  # Positive but low margin
                    score += 5
                valid_metrics += 1
            
            # Operating Margin - 20 points
            if metrics.operating_margin is not None and not np.isnan(metrics.operating_margin):
                op_margin_pct = metrics.operating_margin * 100 if metrics.operating_margin < 1 else metrics.operating_margin
                if op_margin_pct >= 20:  # Excellent operating efficiency
                    score += 20
                elif op_margin_pct >= 12:  # Good operating efficiency
                    score += 15
                elif op_margin_pct >= 5:  # Average operating efficiency
                    score += 8
                elif op_margin_pct > 0:  # Positive but low efficiency
                    score += 3
                valid_metrics += 1
            
            # If no valid metrics, return neutral score
            if valid_metrics == 0:
                return 50.0
            
            # Scale score based on available metrics
            max_possible = 100
            if valid_metrics < 4:
                score = score * (4 / valid_metrics)
            
            return min(100.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating profitability score: {str(e)}")
            return 50.0
    
    def _calculate_valuation_score(self, metrics: FundamentalMetrics, benchmarks: Dict) -> float:
        """Calculate valuation-based score (lower P/E and P/B are generally better)"""
        try:
            score = 0
            valid_metrics = 0
            
            # P/E Ratio - 40 points (lower is better for value)
            if metrics.pe_ratio is not None and not np.isnan(metrics.pe_ratio) and metrics.pe_ratio > 0:
                pe = metrics.pe_ratio
                if pe <= benchmarks['pe_ratio']['good']:
                    score += 40
                elif pe <= benchmarks['pe_ratio']['average']:
                    score += 25
                elif pe <= benchmarks['pe_ratio']['poor']:
                    score += 10
                else:
                    score += 2  # Very high P/E, but not zero points
                valid_metrics += 1
            
            # P/B Ratio - 30 points (lower is generally better)
            if metrics.pb_ratio is not None and not np.isnan(metrics.pb_ratio) and metrics.pb_ratio > 0:
                pb = metrics.pb_ratio
                if pb <= benchmarks['pb_ratio']['good']:
                    score += 30
                elif pb <= benchmarks['pb_ratio']['average']:
                    score += 20
                elif pb <= benchmarks['pb_ratio']['poor']:
                    score += 10
                else:
                    score += 2
                valid_metrics += 1
            
            # EV/EBITDA - 30 points
            if metrics.ev_ebitda is not None and not np.isnan(metrics.ev_ebitda) and metrics.ev_ebitda > 0:
                ev_ebitda = metrics.ev_ebitda
                if ev_ebitda <= 10:  # Good value
                    score += 30
                elif ev_ebitda <= 15:  # Average value
                    score += 20
                elif ev_ebitda <= 25:  # High but acceptable
                    score += 10
                else:
                    score += 2  # Very high valuation
                valid_metrics += 1
            
            # If no valid metrics, return neutral score
            if valid_metrics == 0:
                return 50.0
            
            # Scale score based on available metrics
            if valid_metrics < 3:
                score = score * (3 / valid_metrics)
            
            return min(100.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating valuation score: {str(e)}")
            return 50.0
    
    def _calculate_growth_score(self, metrics: FundamentalMetrics, benchmarks: Dict) -> float:
        """Calculate growth-based score"""
        try:
            score = 0
            valid_metrics = 0
            
            # Revenue Growth - 50 points
            if metrics.revenue_growth is not None and not np.isnan(metrics.revenue_growth):
                revenue_growth_pct = metrics.revenue_growth * 100 if metrics.revenue_growth < 1 else metrics.revenue_growth
                
                if revenue_growth_pct >= benchmarks['revenue_growth']['good']:
                    score += 50
                elif revenue_growth_pct >= benchmarks['revenue_growth']['average']:
                    score += 35
                elif revenue_growth_pct >= benchmarks['revenue_growth']['poor']:
                    score += 20
                elif revenue_growth_pct > 0:
                    score += 10  # Positive but slow growth
                # Negative growth gets 0 points
                valid_metrics += 1
            
            # Earnings Growth - 50 points
            if metrics.earnings_growth is not None and not np.isnan(metrics.earnings_growth):
                earnings_growth_pct = metrics.earnings_growth * 100 if metrics.earnings_growth < 1 else metrics.earnings_growth
                
                if earnings_growth_pct >= 20:  # Excellent earnings growth
                    score += 50
                elif earnings_growth_pct >= 10:  # Good earnings growth
                    score += 35
                elif earnings_growth_pct >= 5:  # Average earnings growth
                    score += 20
                elif earnings_growth_pct > 0:  # Slow but positive growth
                    score += 10
                # Negative earnings growth gets 0 points
                valid_metrics += 1
            
            # If no valid metrics, return neutral score
            if valid_metrics == 0:
                return 50.0
            
            # Scale score based on available metrics
            if valid_metrics < 2:
                score = score * (2 / valid_metrics)
            
            return min(100.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating growth score: {str(e)}")
            return 50.0
    
    def _calculate_financial_health_score(self, metrics: FundamentalMetrics, benchmarks: Dict) -> float:
        """Calculate financial health score"""
        try:
            score = 0
            valid_metrics = 0
            
            # Debt-to-Equity Ratio - 35 points (lower is better)
            if metrics.debt_to_equity is not None and not np.isnan(metrics.debt_to_equity):
                de_ratio = metrics.debt_to_equity
                if de_ratio <= benchmarks['debt_to_equity']['good']:
                    score += 35
                elif de_ratio <= benchmarks['debt_to_equity']['average']:
                    score += 25
                elif de_ratio <= benchmarks['debt_to_equity']['poor']:
                    score += 10
                else:
                    score += 2  # Very high debt
                valid_metrics += 1
            
            # Current Ratio - 35 points (measures liquidity)
            if metrics.current_ratio is not None and not np.isnan(metrics.current_ratio):
                cr = metrics.current_ratio
                if cr >= 2.0:  # Excellent liquidity
                    score += 35
                elif cr >= 1.5:  # Good liquidity
                    score += 25
                elif cr >= 1.0:  # Acceptable liquidity
                    score += 15
                elif cr >= 0.8:  # Concerning but manageable
                    score += 5
                # Below 0.8 gets 0 points (poor liquidity)
                valid_metrics += 1
            
            # Quick Ratio - 30 points (more stringent liquidity measure)
            if metrics.quick_ratio is not None and not np.isnan(metrics.quick_ratio):
                qr = metrics.quick_ratio
                if qr >= 1.5:  # Excellent quick liquidity
                    score += 30
                elif qr >= 1.0:  # Good quick liquidity
                    score += 22
                elif qr >= 0.8:  # Acceptable quick liquidity
                    score += 12
                elif qr >= 0.5:  # Poor but not critical
                    score += 5
                # Below 0.5 gets 0 points
                valid_metrics += 1
            
            # If no valid metrics, return neutral score
            if valid_metrics == 0:
                return 50.0
            
            # Scale score based on available metrics
            if valid_metrics < 3:
                score = score * (3 / valid_metrics)
            
            return min(100.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating financial health score: {str(e)}")
            return 50.0
    
    def get_fundamental_reasons(self, metrics: FundamentalMetrics, score: float, industry: str = None) -> List[str]:
        """Generate reasons for the fundamental score"""
        reasons = []
        
        try:
            benchmarks = self.industry_benchmarks.get(industry, self.industry_benchmarks['default'])
            
            # Profitability reasons
            if metrics.roe is not None and not np.isnan(metrics.roe):
                roe_pct = metrics.roe * 100 if metrics.roe < 1 else metrics.roe
                if roe_pct >= benchmarks['roe']['good']:
                    reasons.append(f"Strong ROE of {roe_pct:.1f}%")
                elif roe_pct < benchmarks['roe']['poor']:
                    reasons.append(f"Low ROE of {roe_pct:.1f}%")
            
            # Valuation reasons
            if metrics.pe_ratio is not None and not np.isnan(metrics.pe_ratio) and metrics.pe_ratio > 0:
                if metrics.pe_ratio <= benchmarks['pe_ratio']['good']:
                    reasons.append(f"Attractive P/E ratio of {metrics.pe_ratio:.1f}")
                elif metrics.pe_ratio > benchmarks['pe_ratio']['poor']:
                    reasons.append(f"High P/E ratio of {metrics.pe_ratio:.1f}")
            
            # Growth reasons
            if metrics.revenue_growth is not None and not np.isnan(metrics.revenue_growth):
                growth_pct = metrics.revenue_growth * 100 if metrics.revenue_growth < 1 else metrics.revenue_growth
                if growth_pct >= benchmarks['revenue_growth']['good']:
                    reasons.append(f"Strong revenue growth of {growth_pct:.1f}%")
                elif growth_pct <= 0:
                    reasons.append(f"Declining revenue ({growth_pct:.1f}%)")
            
            # Financial health reasons
            if metrics.debt_to_equity is not None and not np.isnan(metrics.debt_to_equity):
                if metrics.debt_to_equity <= benchmarks['debt_to_equity']['good']:
                    reasons.append(f"Low debt-to-equity ratio ({metrics.debt_to_equity:.2f})")
                elif metrics.debt_to_equity > benchmarks['debt_to_equity']['poor']:
                    reasons.append(f"High debt-to-equity ratio ({metrics.debt_to_equity:.2f})")
            
            if metrics.current_ratio is not None and not np.isnan(metrics.current_ratio):
                if metrics.current_ratio >= 2.0:
                    reasons.append(f"Strong liquidity (Current ratio: {metrics.current_ratio:.2f})")
                elif metrics.current_ratio < 1.0:
                    reasons.append(f"Poor liquidity (Current ratio: {metrics.current_ratio:.2f})")
            
            # Profit margins
            if metrics.profit_margin is not None and not np.isnan(metrics.profit_margin):
                margin_pct = metrics.profit_margin * 100 if metrics.profit_margin < 1 else metrics.profit_margin
                if margin_pct >= 15:
                    reasons.append(f"Excellent profit margin ({margin_pct:.1f}%)")
                elif margin_pct < 2:
                    reasons.append(f"Low profit margin ({margin_pct:.1f}%)")
            
            # Overall assessment
            if score >= 75:
                reasons.insert(0, "Strong fundamental metrics")
            elif score <= 30:
                reasons.insert(0, "Weak fundamental metrics")
            else:
                reasons.insert(0, "Mixed fundamental signals")
                
            # If no specific reasons found, add generic one
            if len(reasons) == 1:  # Only the overall assessment
                reasons.append("Fundamental analysis completed")
            
        except Exception as e:
            logger.error(f"Error generating fundamental reasons: {str(e)}")
            reasons = ["Fundamental analysis completed"]
        
        return reasons[:5]  # Limit to 5 reasons
    
    def get_risk_assessment(self, metrics: FundamentalMetrics) -> Dict[str, float]:
        """
        Assess fundamental risk factors
        
        Returns:
            Dictionary with risk scores (0-100, higher = more risky)
        """
        try:
            risk_scores = {
                'liquidity_risk': 50.0,
                'leverage_risk': 50.0,
                'profitability_risk': 50.0,
                'valuation_risk': 50.0
            }
            
            # Liquidity risk
            if metrics.current_ratio is not None and not np.isnan(metrics.current_ratio):
                if metrics.current_ratio >= 2.0:
                    risk_scores['liquidity_risk'] = 20.0
                elif metrics.current_ratio >= 1.5:
                    risk_scores['liquidity_risk'] = 35.0
                elif metrics.current_ratio >= 1.0:
                    risk_scores['liquidity_risk'] = 50.0
                elif metrics.current_ratio >= 0.8:
                    risk_scores['liquidity_risk'] = 70.0
                else:
                    risk_scores['liquidity_risk'] = 90.0
            
            # Leverage risk
            if metrics.debt_to_equity is not None and not np.isnan(metrics.debt_to_equity):
                if metrics.debt_to_equity <= 0.3:
                    risk_scores['leverage_risk'] = 20.0
                elif metrics.debt_to_equity <= 0.6:
                    risk_scores['leverage_risk'] = 35.0
                elif metrics.debt_to_equity <= 1.0:
                    risk_scores['leverage_risk'] = 50.0
                elif metrics.debt_to_equity <= 1.5:
                    risk_scores['leverage_risk'] = 70.0
                else:
                    risk_scores['leverage_risk'] = 90.0
            
            # Profitability risk
            if metrics.roe is not None and not np.isnan(metrics.roe):
                roe_pct = metrics.roe * 100 if metrics.roe < 1 else metrics.roe
                if roe_pct >= 15:
                    risk_scores['profitability_risk'] = 25.0
                elif roe_pct >= 10:
                    risk_scores['profitability_risk'] = 40.0
                elif roe_pct >= 5:
                    risk_scores['profitability_risk'] = 60.0
                elif roe_pct > 0:
                    risk_scores['profitability_risk'] = 75.0
                else:
                    risk_scores['profitability_risk'] = 95.0
            
            # Valuation risk (high P/E can be risky)
            if metrics.pe_ratio is not None and not np.isnan(metrics.pe_ratio) and metrics.pe_ratio > 0:
                if metrics.pe_ratio <= 15:
                    risk_scores['valuation_risk'] = 30.0
                elif metrics.pe_ratio <= 25:
                    risk_scores['valuation_risk'] = 45.0
                elif metrics.pe_ratio <= 40:
                    risk_scores['valuation_risk'] = 60.0
                elif metrics.pe_ratio <= 60:
                    risk_scores['valuation_risk'] = 75.0
                else:
                    risk_scores['valuation_risk'] = 90.0
            
            return risk_scores
            
        except Exception as e:
            logger.error(f"Error calculating risk assessment: {str(e)}")
            return {'liquidity_risk': 50.0, 'leverage_risk': 50.0, 
                   'profitability_risk': 50.0, 'valuation_risk': 50.0}
    
    def compare_peer_group(self, metrics_list: List[FundamentalMetrics]) -> Dict[str, Dict]:
        """
        Compare a stock's fundamentals against its peer group
        
        Args:
            metrics_list: List of FundamentalMetrics for peer comparison
            
        Returns:
            Dictionary with peer comparison results
        """
        try:
            if len(metrics_list) < 2:
                return {}
            
            # Extract metrics for comparison
            comparison_data = {}
            
            for metric_name in ['pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity', 
                               'current_ratio', 'profit_margin', 'revenue_growth']:
                values = []
                symbols = []
                
                for metrics in metrics_list:
                    value = getattr(metrics, metric_name, None)
                    if value is not None and not np.isnan(value):
                        values.append(value)
                        symbols.append(metrics.symbol)
                
                if len(values) >= 2:
                    comparison_data[metric_name] = {
                        'values': dict(zip(symbols, values)),
                        'median': np.median(values),
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
            
            # Rank companies
            rankings = {}
            for metrics in metrics_list:
                symbol = metrics.symbol
                rankings[symbol] = {'total_rank': 0, 'metric_ranks': {}}
                
                for metric_name, data in comparison_data.items():
                    if symbol in data['values']:
                        value = data['values'][symbol]
                        
                        # Determine if higher or lower is better
                        if metric_name in ['pe_ratio', 'pb_ratio', 'debt_to_equity']:
                            # Lower is better
                            rank = sum(1 for v in data['values'].values() if v > value)
                        else:
                            # Higher is better
                            rank = sum(1 for v in data['values'].values() if v < value)
                        
                        rankings[symbol]['metric_ranks'][metric_name] = rank
                        rankings[symbol]['total_rank'] += rank
            
            return {
                'comparison_data': comparison_data,
                'rankings': rankings
            }
            
        except Exception as e:
            logger.error(f"Error in peer group comparison: {str(e)}")
            return {}

"""
Bankroll Management System for BetFinder AI
Tracks betting history, calculates ROI, manages risk, and provides insights
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

class BankrollManager:
    def __init__(self, data_file: str = "bankroll_data.json"):
        self.data_file = data_file
        self.data = self.load_data()
        
    def load_data(self) -> Dict:
        """Load bankroll data from file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default data structure
        return {
            "starting_bankroll": 1000.0,
            "current_bankroll": 1000.0,
            "unit_size": 10.0,  # 1% of starting bankroll
            "max_bet_percentage": 5.0,  # Max 5% of bankroll per bet
            "bets": [],
            "settings": {
                "conservative_mode": True,
                "max_daily_bets": 5,
                "max_daily_risk": 15.0,  # Max 15% of bankroll per day
                "stop_loss_percentage": 20.0,  # Stop when down 20%
                "profit_target_percentage": 50.0  # Target 50% profit
            },
            "daily_stats": {},
            "monthly_stats": {}
        }
    
    def save_data(self):
        """Save bankroll data to file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def initialize_bankroll(self, starting_amount: float, unit_size: float = None):
        """Initialize or reset bankroll"""
        self.data["starting_bankroll"] = starting_amount
        self.data["current_bankroll"] = starting_amount
        if unit_size:
            self.data["unit_size"] = unit_size
        else:
            self.data["unit_size"] = starting_amount * 0.01  # 1% default
        self.save_data()
    
    def calculate_recommended_bet_size(self, confidence: float, expected_value: float, 
                                     kelly_fraction: float = 0.25) -> Tuple[float, str]:
        """Calculate optimal bet size using Kelly Criterion with constraints"""
        current_bankroll = self.data["current_bankroll"]
        unit_size = self.data["unit_size"]
        max_bet_pct = self.data["max_bet_percentage"] / 100
        
        # Kelly Criterion calculation (conservative)
        if expected_value <= 0:
            return 0.0, "No bet recommended (negative EV)"
        
        # Convert confidence to probability
        probability = confidence / 100
        
        # Kelly fraction (conservative approach)
        kelly_bet = kelly_fraction * (expected_value / 100) * current_bankroll
        
        # Apply constraints
        max_bet = current_bankroll * max_bet_pct
        unit_based_bet = (confidence / 100) * 3 * unit_size  # Max 3 units based on confidence
        
        # Use the minimum of all constraints
        recommended_bet = min(kelly_bet, max_bet, unit_based_bet)
        
        # Check daily limits
        today = datetime.now().strftime("%Y-%m-%d")
        daily_risk = self.get_daily_risk_used(today)
        daily_limit = current_bankroll * (self.data["settings"]["max_daily_risk"] / 100)
        
        if daily_risk + recommended_bet > daily_limit:
            remaining_daily = max(0, daily_limit - daily_risk)
            recommended_bet = min(recommended_bet, remaining_daily)
            if recommended_bet <= 0:
                return 0.0, f"Daily risk limit reached (${daily_risk:.2f}/${daily_limit:.2f})"
        
        # Minimum bet check
        if recommended_bet < unit_size * 0.5:
            return 0.0, "Bet size too small relative to unit size"
        
        # Calculate as units
        units = recommended_bet / unit_size
        
        return recommended_bet, f"{units:.1f} units (${recommended_bet:.2f})"
    
    def get_daily_risk_used(self, date: str) -> float:
        """Get total amount at risk for a specific date"""
        total_risk = 0.0
        for bet in self.data["bets"]:
            if bet["date"] == date and bet["status"] in ["pending", "active"]:
                total_risk += bet["bet_amount"]
        return total_risk
    
    def add_bet(self, pick_id: str, sport: str, matchup: str, pick_type: str, 
                odds: int, bet_amount: float, confidence: float, expected_value: float) -> bool:
        """Add a new bet to tracking"""
        if bet_amount <= 0:
            return False
        
        # Check if enough bankroll
        if bet_amount > self.data["current_bankroll"]:
            return False
        
        bet = {
            "id": f"{pick_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "sport": sport,
            "matchup": matchup,
            "pick_type": pick_type,
            "odds": odds,
            "bet_amount": bet_amount,
            "potential_payout": self.calculate_payout(bet_amount, odds),
            "confidence": confidence,
            "expected_value": expected_value,
            "status": "pending",  # pending, won, lost, pushed
            "result_amount": 0.0,
            "notes": ""
        }
        
        self.data["bets"].append(bet)
        self.update_daily_stats()
        self.save_data()
        return True
    
    def calculate_payout(self, bet_amount: float, odds: int) -> float:
        """Calculate potential payout from odds"""
        if odds > 0:
            return bet_amount * (odds / 100) + bet_amount
        else:
            return bet_amount * (100 / abs(odds)) + bet_amount
    
    def settle_bet(self, bet_id: str, status: str, result_amount: float = None):
        """Settle a bet (won, lost, pushed)"""
        for bet in self.data["bets"]:
            if bet["id"] == bet_id:
                bet["status"] = status
                
                if status == "won":
                    bet["result_amount"] = result_amount or bet["potential_payout"]
                    self.data["current_bankroll"] += bet["result_amount"] - bet["bet_amount"]
                elif status == "lost":
                    bet["result_amount"] = 0.0
                    self.data["current_bankroll"] -= bet["bet_amount"]
                elif status == "pushed":
                    bet["result_amount"] = bet["bet_amount"]
                
                self.update_daily_stats()
                self.update_monthly_stats()
                self.save_data()
                break
    
    def update_daily_stats(self):
        """Update daily statistics"""
        today = datetime.now().strftime("%Y-%m-%d")
        today_bets = [bet for bet in self.data["bets"] if bet["date"] == today]
        
        pending_bets = [bet for bet in today_bets if bet["status"] == "pending"]
        settled_bets = [bet for bet in today_bets if bet["status"] in ["won", "lost", "pushed"]]
        
        self.data["daily_stats"][today] = {
            "total_bets": len(today_bets),
            "pending_bets": len(pending_bets),
            "settled_bets": len(settled_bets),
            "total_wagered": sum(bet["bet_amount"] for bet in today_bets),
            "potential_payout": sum(bet["potential_payout"] for bet in pending_bets),
            "actual_return": sum(bet["result_amount"] for bet in settled_bets),
            "net_profit": sum(bet["result_amount"] - bet["bet_amount"] for bet in settled_bets),
            "wins": len([bet for bet in settled_bets if bet["status"] == "won"]),
            "losses": len([bet for bet in settled_bets if bet["status"] == "lost"]),
            "pushes": len([bet for bet in settled_bets if bet["status"] == "pushed"])
        }
    
    def update_monthly_stats(self):
        """Update monthly statistics"""
        current_month = datetime.now().strftime("%Y-%m")
        month_bets = [bet for bet in self.data["bets"] 
                     if bet["date"].startswith(current_month) and bet["status"] in ["won", "lost", "pushed"]]
        
        if month_bets:
            total_wagered = sum(bet["bet_amount"] for bet in month_bets)
            total_return = sum(bet["result_amount"] for bet in month_bets)
            net_profit = total_return - total_wagered
            
            self.data["monthly_stats"][current_month] = {
                "total_bets": len(month_bets),
                "total_wagered": total_wagered,
                "total_return": total_return,
                "net_profit": net_profit,
                "roi_percentage": (net_profit / total_wagered * 100) if total_wagered > 0 else 0,
                "win_rate": len([bet for bet in month_bets if bet["status"] == "won"]) / len(month_bets) * 100,
                "avg_bet_size": total_wagered / len(month_bets),
                "biggest_win": max([bet["result_amount"] - bet["bet_amount"] for bet in month_bets if bet["status"] == "won"] or [0]),
                "biggest_loss": min([bet["result_amount"] - bet["bet_amount"] for bet in month_bets if bet["status"] == "lost"] or [0])
            }
    
    def get_performance_metrics(self) -> Dict:
        """Get overall performance metrics"""
        all_settled_bets = [bet for bet in self.data["bets"] if bet["status"] in ["won", "lost", "pushed"]]
        
        if not all_settled_bets:
            return {"total_bets": 0, "message": "No settled bets yet"}
        
        total_wagered = sum(bet["bet_amount"] for bet in all_settled_bets)
        total_return = sum(bet["result_amount"] for bet in all_settled_bets)
        net_profit = total_return - total_wagered
        
        wins = [bet for bet in all_settled_bets if bet["status"] == "won"]
        losses = [bet for bet in all_settled_bets if bet["status"] == "lost"]
        
        return {
            "total_bets": len(all_settled_bets),
            "total_wagered": total_wagered,
            "total_return": total_return,
            "net_profit": net_profit,
            "roi_percentage": (net_profit / total_wagered * 100) if total_wagered > 0 else 0,
            "win_rate": len(wins) / len(all_settled_bets) * 100,
            "current_bankroll": self.data["current_bankroll"],
            "bankroll_change": self.data["current_bankroll"] - self.data["starting_bankroll"],
            "bankroll_change_pct": ((self.data["current_bankroll"] - self.data["starting_bankroll"]) / self.data["starting_bankroll"] * 100),
            "avg_bet_size": total_wagered / len(all_settled_bets),
            "biggest_win": max([bet["result_amount"] - bet["bet_amount"] for bet in wins] or [0]),
            "biggest_loss": min([bet["result_amount"] - bet["bet_amount"] for bet in losses] or [0]),
            "current_streak": self.get_current_streak(),
            "avg_odds": sum(bet["odds"] for bet in all_settled_bets) / len(all_settled_bets)
        }
    
    def get_current_streak(self) -> Dict:
        """Get current winning/losing streak"""
        recent_bets = sorted([bet for bet in self.data["bets"] if bet["status"] in ["won", "lost"]], 
                           key=lambda x: f"{x['date']} {x['time']}", reverse=True)
        
        if not recent_bets:
            return {"type": "none", "count": 0}
        
        streak_type = recent_bets[0]["status"]
        streak_count = 1
        
        for bet in recent_bets[1:]:
            if bet["status"] == streak_type:
                streak_count += 1
            else:
                break
        
        return {"type": streak_type, "count": streak_count}
    
    def get_risk_assessment(self) -> Dict:
        """Assess current risk levels"""
        current_bankroll = self.data["current_bankroll"]
        starting_bankroll = self.data["starting_bankroll"]
        settings = self.data["settings"]
        
        # Calculate drawdown
        max_bankroll = max([starting_bankroll] + [bet.get("bankroll_after", starting_bankroll) for bet in self.data["bets"]])
        current_drawdown = (max_bankroll - current_bankroll) / max_bankroll * 100
        
        # Daily risk assessment
        today = datetime.now().strftime("%Y-%m-%d")
        daily_risk = self.get_daily_risk_used(today)
        daily_limit = current_bankroll * (settings["max_daily_risk"] / 100)
        daily_risk_pct = (daily_risk / current_bankroll * 100) if current_bankroll > 0 else 0
        
        # Risk status
        risk_level = "Low"
        if current_drawdown > 15 or daily_risk_pct > 10:
            risk_level = "High"
        elif current_drawdown > 10 or daily_risk_pct > 7:
            risk_level = "Medium"
        
        return {
            "risk_level": risk_level,
            "current_drawdown": current_drawdown,
            "daily_risk_used": daily_risk,
            "daily_risk_limit": daily_limit,
            "daily_risk_percentage": daily_risk_pct,
            "stop_loss_triggered": current_drawdown >= settings["stop_loss_percentage"],
            "profit_target_reached": ((current_bankroll - starting_bankroll) / starting_bankroll * 100) >= settings["profit_target_percentage"],
            "recommendations": self.get_risk_recommendations(current_drawdown, daily_risk_pct)
        }
    
    def get_risk_recommendations(self, drawdown: float, daily_risk_pct: float) -> List[str]:
        """Get risk management recommendations"""
        recommendations = []
        
        if drawdown > 15:
            recommendations.append("🚨 High drawdown detected - consider reducing bet sizes")
        elif drawdown > 10:
            recommendations.append("⚠️ Moderate drawdown - review recent bet quality")
        
        if daily_risk_pct > 10:
            recommendations.append("🚨 High daily risk exposure - consider stopping for today")
        elif daily_risk_pct > 7:
            recommendations.append("⚠️ Approaching daily risk limit")
        
        if len(self.data["bets"]) > 0:
            recent_performance = self.get_recent_performance(7)  # Last 7 days
            if recent_performance["win_rate"] < 40:
                recommendations.append("📉 Recent performance below expectations - review strategy")
        
        if not recommendations:
            recommendations.append("✅ Risk levels are within acceptable parameters")
        
        return recommendations
    
    def get_recent_performance(self, days: int) -> Dict:
        """Get performance over recent days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_bets = [bet for bet in self.data["bets"] 
                      if datetime.strptime(bet["date"], "%Y-%m-%d") >= cutoff_date 
                      and bet["status"] in ["won", "lost", "pushed"]]
        
        if not recent_bets:
            return {"total_bets": 0, "win_rate": 0, "roi": 0}
        
        total_wagered = sum(bet["bet_amount"] for bet in recent_bets)
        total_return = sum(bet["result_amount"] for bet in recent_bets)
        wins = len([bet for bet in recent_bets if bet["status"] == "won"])
        
        return {
            "total_bets": len(recent_bets),
            "win_rate": wins / len(recent_bets) * 100,
            "roi": ((total_return - total_wagered) / total_wagered * 100) if total_wagered > 0 else 0,
            "net_profit": total_return - total_wagered
        }
    
    def get_bet_history_dataframe(self) -> pd.DataFrame:
        """Get bet history as pandas DataFrame for analysis"""
        if not self.data["bets"]:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.data["bets"])
        df["date"] = pd.to_datetime(df["date"])
        df["profit"] = df["result_amount"] - df["bet_amount"]
        df["roi"] = (df["profit"] / df["bet_amount"] * 100).round(2)
        
        return df
    
    def export_data(self, format: str = "csv") -> str:
        """Export betting data for external analysis"""
        df = self.get_bet_history_dataframe()
        
        if df.empty:
            return "No data to export"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "csv":
            filename = f"betting_history_{timestamp}.csv"
            df.to_csv(filename, index=False)
            return filename
        elif format == "json":
            filename = f"betting_data_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(self.data, f, indent=2, default=str)
            return filename
        
        return "Unsupported format"

# Example usage and testing
if __name__ == "__main__":
    manager = BankrollManager()
    
    # Initialize bankroll
    manager.initialize_bankroll(1000.0, 10.0)
    
    # Test bet sizing
    bet_size, reason = manager.calculate_recommended_bet_size(75, 5.2)
    print(f"Recommended bet: {reason}")
    
    # Add a sample bet
    success = manager.add_bet(
        pick_id="test_001",
        sport="basketball",
        matchup="Lakers vs Warriors",
        pick_type="moneyline",
        odds=-110,
        bet_amount=bet_size,
        confidence=75,
        expected_value=5.2
    )
    
    print(f"Bet added: {success}")
    
    # Get performance metrics
    metrics = manager.get_performance_metrics()
    print("Performance metrics:", metrics)
    
    # Get risk assessment
    risk = manager.get_risk_assessment()
    print("Risk assessment:", risk)
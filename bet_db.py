import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional

class BetDatabase:
    def __init__(self, db_path: str = 'bets.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create bets table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_name TEXT NOT NULL,
                    sport TEXT NOT NULL,
                    bet_type TEXT NOT NULL,
                    odds REAL NOT NULL,
                    stake REAL NOT NULL,
                    potential_payout REAL NOT NULL,
                    bookmaker TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    settled_at TIMESTAMP,
                    result TEXT
                )
            ''')
            
            # Create odds_comparison table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS odds_comparison (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_name TEXT NOT NULL,
                    sport TEXT NOT NULL,
                    bookmaker TEXT NOT NULL,
                    bet_type TEXT NOT NULL,
                    odds REAL NOT NULL,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create analysis_results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_name TEXT NOT NULL,
                    sport TEXT NOT NULL,
                    analysis_data TEXT,
                    recommendation TEXT,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def add_bet(self, event_name: str, sport: str, bet_type: str, 
                odds: float, stake: float, bookmaker: str) -> int:
        """Add a new bet to the database."""
        potential_payout = stake * odds
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO bets (event_name, sport, bet_type, odds, stake, 
                                potential_payout, bookmaker)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (event_name, sport, bet_type, odds, stake, potential_payout, bookmaker))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_bets(self, status: Optional[str] = None) -> List[Dict]:
        """Retrieve bets from the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if status:
                cursor.execute('SELECT * FROM bets WHERE status = ? ORDER BY created_at DESC', (status,))
            else:
                cursor.execute('SELECT * FROM bets ORDER BY created_at DESC')
            
            return [dict(row) for row in cursor.fetchall()]
    
    def update_bet_status(self, bet_id: int, status: str, result: Optional[str] = None):
        """Update bet status and result."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if result:
                cursor.execute('''
                    UPDATE bets 
                    SET status = ?, result = ?, settled_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (status, result, bet_id))
            else:
                cursor.execute('UPDATE bets SET status = ? WHERE id = ?', (status, bet_id))
            
            conn.commit()
    
    def add_odds_comparison(self, event_name: str, sport: str, bookmaker: str,
                           bet_type: str, odds: float):
        """Add odds data for comparison."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO odds_comparison (event_name, sport, bookmaker, bet_type, odds)
                VALUES (?, ?, ?, ?, ?)
            ''', (event_name, sport, bookmaker, bet_type, odds))
            
            conn.commit()
    
    def get_best_odds(self, event_name: str, bet_type: str) -> List[Dict]:
        """Get the best odds for a specific event and bet type."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT bookmaker, odds, scraped_at
                FROM odds_comparison 
                WHERE event_name = ? AND bet_type = ?
                ORDER BY odds DESC
            ''', (event_name, bet_type))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def save_analysis_result(self, event_name: str, sport: str, 
                           analysis_data: Dict, recommendation: str,
                           confidence_score: float):
        """Save AI analysis results."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO analysis_results 
                (event_name, sport, analysis_data, recommendation, confidence_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (event_name, sport, json.dumps(analysis_data), 
                  recommendation, confidence_score))
            
            conn.commit()
    
    def get_analysis_results(self, event_name: Optional[str] = None) -> List[Dict]:
        """Get analysis results from the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if event_name:
                cursor.execute('''
                    SELECT * FROM analysis_results 
                    WHERE event_name = ? 
                    ORDER BY created_at DESC
                ''', (event_name,))
            else:
                cursor.execute('SELECT * FROM analysis_results ORDER BY created_at DESC')
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result['analysis_data']:
                    result['analysis_data'] = json.loads(result['analysis_data'])
                results.append(result)
            
            return results
    
    def get_betting_stats(self) -> Dict:
        """Get overall betting statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total bets
            cursor.execute('SELECT COUNT(*) FROM bets')
            total_bets = cursor.fetchone()[0]
            
            # Total stake
            cursor.execute('SELECT SUM(stake) FROM bets')
            total_stake = cursor.fetchone()[0] or 0
            
            # Won bets
            cursor.execute('SELECT COUNT(*) FROM bets WHERE result = "won"')
            won_bets = cursor.fetchone()[0]
            
            # Lost bets
            cursor.execute('SELECT COUNT(*) FROM bets WHERE result = "lost"')
            lost_bets = cursor.fetchone()[0]
            
            # Total winnings
            cursor.execute('SELECT SUM(potential_payout) FROM bets WHERE result = "won"')
            total_winnings = cursor.fetchone()[0] or 0
            
            # Calculate win rate
            win_rate = (won_bets / total_bets * 100) if total_bets > 0 else 0
            
            # Calculate profit/loss
            total_lost_stake = cursor.execute('SELECT SUM(stake) FROM bets WHERE result = "lost"').fetchone()[0] or 0
            profit_loss = total_winnings - total_lost_stake
            
            return {
                'total_bets': total_bets,
                'total_stake': round(total_stake, 2),
                'won_bets': won_bets,
                'lost_bets': lost_bets,
                'win_rate': round(win_rate, 2),
                'total_winnings': round(total_winnings, 2),
                'profit_loss': round(profit_loss, 2)
            }

# Initialize database instance
db = BetDatabase()

if __name__ == '__main__':
    # Test the database functionality
    print("Testing BetDatabase...")
    
    # Add sample bet
    bet_id = db.add_bet(
        event_name="Manchester United vs Chelsea",
        sport="Football",
        bet_type="1X2 - Home Win",
        odds=2.50,
        stake=20.0,
        bookmaker="Bet365"
    )
    print(f"Added bet with ID: {bet_id}")
    
    # Get all bets
    bets = db.get_bets()
    print(f"Total bets: {len(bets)}")
    
    # Get betting stats
    stats = db.get_betting_stats()
    print(f"Betting stats: {stats}")
    
    print("Database test completed!")

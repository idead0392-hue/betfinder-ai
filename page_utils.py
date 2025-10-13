import os
import time
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as components_html

# Failsafe: Remove any rendered literal </div> text from the DOM
components_html("""
<script>
function removeStrayDivText() {
    // Remove any text nodes with stray closing tags
    const container = document.querySelector('[data-testid=\"stAppViewContainer\"]');
    if (!container) return;
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null, false);
    const nodes = [];
    let n;
    while (n = walker.nextNode()) {
        if (n.textContent && n.textContent.trim().includes("</div>")) {
            nodes.push(n);
        }
    }
    nodes.forEach(t => {
        if (t.parentNode) t.parentNode.removeChild(t);
    });
}
// Run repeatedly for dynamic content (covers tabs)
setInterval(removeStrayDivText, 400);
</script>
""", height=0)
from functools import lru_cache


def get_effective_csv_path() -> str:
    if 'prizepicks_csv_path' not in st.session_state:
        st.session_state['prizepicks_csv_path'] = 'prizepicks_props.csv'
    return st.session_state['prizepicks_csv_path']


def ensure_fresh_csv(path: str, max_age_sec: int = 300, target_sport: str | None = None) -> None:
    """
    Ensure the CSV is fresh, with a lightweight cross-process lock to avoid
    multiple concurrent scrapes triggered by multiple tabs/pages.
    """
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = 0
    age = time.time() - mtime if mtime else 1e9

    # If the file is fresh within the configured interval, skip re-scrape
    # Keep a small minimum window to avoid thrashing if interval is very low
    if age <= max(15, max_age_sec):
        return

    lock_path = f"{path}.lock"
    lock_acquired = False
    try:
        # Try to acquire a simple PID-based lock
        if not os.path.exists(lock_path):
            with open(lock_path, 'w') as f:
                f.write(str(os.getpid()))
            lock_acquired = True
        else:
            # If lock is stale (>3 minutes), override
            try:
                lmtime = os.path.getmtime(lock_path)
            except Exception:
                lmtime = 0
            if (time.time() - lmtime) > 180:
                with open(lock_path, 'w') as f:
                    f.write(str(os.getpid()))
                lock_acquired = True

        if lock_acquired:
            os.environ['PRIZEPICKS_CSV'] = path
            # Hint the scraper to fetch a specific sport only (best-effort)
            if target_sport:
                sport_hint_map = {
                    'basketball': 'nba',
                    'football': 'nfl',
                    'college_football': 'cfb',
                    'baseball': 'mlb',
                    'hockey': 'nhl',
                    'soccer': 'soccer'
                }
                os.environ['PRIZEPICKS_SPORT_PARAM'] = sport_hint_map.get(target_sport, '')
            from prizepicks_scrape import main as scrape_main
            scrape_main()
    except Exception:
        pass
    finally:
        if lock_acquired:
            try:
                os.remove(lock_path)
            except Exception:
                pass


def _normalize_league_to_sport(value: str) -> str:
    if not value:
        return ''
    s = str(value).strip().lower()
    aliases = {
        # Core traditional sports
        'nba': 'basketball', 'wnba': 'basketball', 'cbb': 'basketball',
        'nfl': 'football', 'cfb': 'college_football', 'ncaa football': 'college_football',
        'mlb': 'baseball', 'nhl': 'hockey', 'epl': 'soccer', 'soccer': 'soccer',
        # Esports titles
        'league of legends': 'league_of_legends', 'lol': 'league_of_legends',
        'valorant': 'valorant', 'valo': 'valorant',
        'dota 2': 'dota2', 'dota2': 'dota2',
        'overwatch': 'overwatch', 'overwatch 2': 'overwatch', 'ow': 'overwatch',
        'rocket league': 'rocket_league', 'rocket_league': 'rocket_league', 'rl': 'rocket_league',
        'csgo': 'csgo', 'cs:go': 'csgo', 'cs2': 'csgo', 'counter-strike': 'csgo', 'counter strike': 'csgo', 'counter-strike 2': 'csgo',
        'apex': 'apex', 'apex legends': 'apex'
    }
    return aliases.get(s, s)


# --------------------
# Strict NHL filtering
# --------------------

NHL_TEAM_NAMES = [
    'bruins', 'sabres', 'red wings', 'panthers', 'canadiens', 'senators', 'lightning', 'maple leafs',
    'hurricanes', 'blue jackets', 'devils', 'islanders', 'rangers', 'flyers', 'penguins', 'capitals',
    'blackhawks', 'avalanche', 'stars', 'wild', 'predators', 'blues', 'jets', 'ducks', 'coyotes',
    'flames', 'oilers', 'kings', 'sharks', 'kraken', 'canucks', 'golden knights'
]

NHL_TEAM_ABBREVS = [
    'bos','buf','det','fla','mtl','ott','tbl','tb','tor','car','cbj','njd','nj','nyi','nyr','phi','pit','wsh','chi','col','dal','min','nsh','stl','wpg','ana','ari','cgy','edm','lak','la','sjs','sj','sea','van','vgk'
]

NHL_HOCKEY_STATS = [
    'goals', 'assists', 'shots on goal', 'sog', 'saves'
]

SOCCER_ONLY_STATS = ['goals conceded', 'yellow cards', 'red cards', 'clean sheets']


@lru_cache(maxsize=1)
def _load_nhl_roster() -> List[str]:
    """Load NHL roster from data file, fallback to a core whitelist of star players.
    The list is lowercase names. To expand coverage, add data/nhl_roster.txt (one name per line)."""
    roster: List[str] = []
    try:
        roster_path = os.path.join(os.getcwd(), 'data', 'nhl_roster.txt')
        if os.path.exists(roster_path):
            with open(roster_path, 'r', encoding='utf-8') as f:
                roster = [line.strip().lower() for line in f if line.strip()]
    except Exception:
        pass
    if not roster:
        roster = [
            'sidney crosby','evgeni malkin','auston matthews','mitch marner','nathan mackinnon','mikko rantanen',
            'connor mcdavid','leon draisaitl','nikita kucherov','brayden point','matthew tkachuk','brady tkachuk',
            'cale makar','adam fox','victor hedman','roman josi','david pastrnak','brad marchand','igor shesterkin',
            'andrei vasilevskiy','ilja sorokin','jason robertson','alex ovechkin','patrick kane','jack eichel',
        ]
    return roster


def _is_team_nhl(team: str, matchup: str) -> bool:
    t = (team or '').lower()
    m = (matchup or '').lower()
    if any(name in t for name in NHL_TEAM_NAMES) or any(ab in t for ab in NHL_TEAM_ABBREVS):
        return True
    if any(name in m for name in NHL_TEAM_NAMES) or any(ab in m for ab in NHL_TEAM_ABBREVS):
        return True
    return False


def hockey_strict_filter(prop: dict) -> bool:
    """Strict NHL prop filter: require NHL team, NHL player (if enabled), and hockey stats only."""
    if not prop:
        return False
    player = str(prop.get('player_name','')).lower()
    team = str(prop.get('team','')).lower()
    stat = str(prop.get('stat_type','')).lower()
    matchup = str(prop.get('matchup','')).lower()
    league = str(prop.get('league','')).lower()

    # League must be NHL when present
    if league and not any(x in league for x in ['nhl','nhl1p']):
        return False

    # Team/matchup must be NHL
    if not _is_team_nhl(team, matchup):
        return False

    # Stat must be allowed hockey type and not soccer-only
    if not any(s in stat for s in NHL_HOCKEY_STATS):
        return False
    if any(s in stat for s in SOCCER_ONLY_STATS):
        return False

    # Optional strict player roster enforcement (requires a reasonably complete roster)
    if os.getenv('STRICT_NHL_PLAYER_FILTER','1') == '1':
        roster = _load_nhl_roster()
        if len(roster) >= 50 and player not in roster:
            return False

    return True


def _log_misclassified(prop: dict, sport: str, reason: str = "") -> None:
    """Append misclassified prop details to a per-sport log file for admin review, including reason."""
    try:
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'misclassified_{sport}.log')
        # Keep it simple: one-line JSON-ish dict for easy tailing
        payload = {
            'player': prop.get('player_name'),
            'team': prop.get('team'),
            'stat': prop.get('stat_type'),
            'league': prop.get('league'),
            'matchup': prop.get('matchup'),
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(str(payload) + "\n")
    except Exception:
        # Best-effort logging only
        pass


def _map_to_sport(p: dict) -> str:

    player = str(p.get('player_name', '')).lower()
    stat = str(p.get('stat_type', '')).lower()
    team = str(p.get('team', '')).lower()
    matchup = str(p.get('matchup', '')).lower()
    orig_sport = str(p.get('sport', '')).lower()

    # --- Sport-specific cheat sheets ---
    football_terms = ["kicking points", "field goals made", "touchdowns", "pass yards", "passing yards", "rush yards", "rushing yards", "receiving yards", "receptions", "field goals", "extra points"]
    basketball_terms = ["points", "rebounds", "assists", "blocks", "steals", "3pt made", "pts+rebs+asts"]
    hockey_terms = ["shots", "penalty minutes", "power play", "faceoff", "time on ice", "plus/minus", "blocked shots", "goalie saves"]
    baseball_terms = ["hits", "home runs", "rbis", "strikeouts", "total bases", "stolen bases"]
    tennis_terms = ["aces", "double faults", "games won", "sets won"]
    soccer_terms = ["goals", "assists", "shots on goal", "shots on target", "goal + assist", "fouls", "cards", "clean sheets", "saves", "goalie saves"]
    
    # Esports-specific terms
    csgo_terms = ["map kills", "map deaths", "map assists", "rounds won", "headshots", "adr", "rating", "maps won"]
    lol_terms = ["creep score", "cs", "vision score", "gold earned", "damage dealt", "towers destroyed", "dragons killed", "barons killed"]
    dota_terms = ["last hits", "denies", "gpm", "xpm", "networth", "tower damage", "healing done", "roshan kills"]

    football_teams = ["patriots", "saints", "packers", "cowboys", "giants", "jets", "bears", "steelers", "eagles", "chiefs", "49ers", "ravens", "bengals", "bills", "dolphins", "broncos", "lions", "jaguars", "panthers", "raiders", "chargers", "seahawks", "buccaneers", "cardinals", "commanders", "colts", "vikings", "texans", "falcons", "rams", "titans"]
    basketball_teams = ["lakers", "knicks", "warriors", "celtics", "bulls", "nets", "suns", "mavericks", "clippers", "spurs", "heat", "bucks", "76ers", "nuggets", "hawks", "pelicans", "grizzlies", "timberwolves", "magic", "wizards", "pacers", "pistons", "hornets", "jazz", "thunder", "raptors", "rockets", "kings", "trail blazers"]

    football_kickers = ["jason myers", "chris boswell", "cam little", "michael badgley", "justin tucker", "harrison butker", "younghoe koo", "daniel carlson", "greg joseph", "matt gay", "brandon mcmanus", "evan mcpherson", "eddie pineiro", "riley patterson", "brett maher", "nick folk", "cairo santos", "dustin hopkins", "jake elliott", "graham gano", "chase mclaughlin", "andrew mevis", "joey slye", "chris naggar", "sam ficken"]

    # 1. Check league/sport mapping first for esports (more reliable than stats)
    aliases = {
        'nba': 'basketball', 'wnba': 'basketball', 'cbb': 'basketball',
        'nfl': 'football', 'cfb': 'college_football', 'ncaa football': 'college_football',
        'mlb': 'baseball', 'nhl': 'hockey', 'epl': 'soccer', 'soccer': 'soccer',
        'league of legends': 'league_of_legends', 'lol': 'league_of_legends', 'league_of_legends': 'league_of_legends',
        'valorant': 'valorant', 'valo': 'valorant',
        'dota 2': 'dota2', 'dota2': 'dota2',
        'overwatch': 'overwatch', 'overwatch 2': 'overwatch', 'ow': 'overwatch',
        'rocket league': 'rocket_league', 'rocket_league': 'rocket_league', 'rl': 'rocket_league',
        'csgo': 'csgo', 'cs:go': 'csgo', 'cs2': 'csgo', 'counter-strike': 'csgo', 'counter strike': 'csgo', 'counter-strike 2': 'csgo',
        'apex': 'apex', 'apex legends': 'apex'
    }
    mapped = aliases.get(orig_sport.lower())
    if mapped:
        return mapped

    # 2. Stat label check (most robust for traditional sports)
    if stat in football_terms:
        return "football"
    if stat in basketball_terms:
        return "basketball"
    if stat in hockey_terms:
        return "hockey"
    if stat in baseball_terms:
        return "baseball"
    if stat in tennis_terms:
        return "tennis"
    if stat in soccer_terms:
        return "soccer"
    
    # Esports stat detection - only if league mapping didn't work
    if any(cs in stat for cs in csgo_terms):
        return "csgo"
    elif "kills" in stat and ("map" in stat or "round" in stat):
        return "csgo"
    elif any(ls in stat for ls in lol_terms):
        return "league_of_legends"
    elif "cs" in stat and "creep" in stat:
        return "league_of_legends"
    elif any(ds in stat for ds in dota_terms):
        return "dota2"
    elif "gpm" in stat or "xpm" in stat:
        return "dota2"

    # 3. Team name check (matchup, team, or player)
    if any(t in team for t in football_teams) or any(t in matchup for t in football_teams):
        return "football"
    if any(t in team for t in basketball_teams) or any(t in matchup for t in basketball_teams):
        return "basketball"

    # 3. Football kicker check
    if stat == "kicking points" and (any(k in player for k in football_kickers) or any(k in team for k in football_teams)):
        return "football"

    # 4. Football kicker check
    if stat == "kicking points" and (any(k in player for k in football_kickers) or any(k in team for k in football_teams)):
        return "football"

    # 5. Legacy player/stat heuristics (for esports, hockey, etc.)
    # ...existing code...

    return ''


def _validate_prop_consistency(prop_data: dict) -> bool:
    """
    Validate that prop data is internally consistent (team, matchup, sport alignment)
    Returns False if the prop should be filtered out due to inconsistencies
    """
    player_name = str(prop_data.get('player_name', '')).lower()
    stat_type = str(prop_data.get('stat_type', '')).lower()
    team = str(prop_data.get('team', '')).lower()
    matchup = str(prop_data.get('matchup', '')).lower()
    league = str(prop_data.get('league', '')).lower()
    
    # Skip props with missing essential data
    if not player_name or not stat_type:
        return False
    
    # Skip if team is completely inconsistent with matchup
    if team and matchup and '@' in matchup:
        # Extract teams from matchup (e.g., "CHI @ WAS" -> ["chi", "was"])
        parts = matchup.split('@')
        if len(parts) == 2:
            away_team = parts[0].strip().lower()
            home_team = parts[1].strip().lower()
            
            # Team should appear in either away or home position
            if team not in away_team and team not in home_team and away_team not in team and home_team not in team:
                # Additional check: see if it's an abbreviation mismatch
                team_abbrevs = {
                    'new england patriots': 'ne', 'new orleans saints': 'no',
                    'chicago bears': 'chi', 'washington commanders': 'was',
                    'carolina panthers': 'car', 'arizona cardinals': 'ari',
                    'indianapolis colts': 'ind'
                }
                
                team_full_name = None
                for full, abbrev in team_abbrevs.items():
                    if abbrev == team or team in full:
                        team_full_name = full
                        break
                
                if team_full_name:
                    # Check if full team name appears in matchup
                    if team_full_name not in matchup:
                        return False
                else:
                    # If no mapping found and team doesn't match, filter out
                    return False
    
    # Additional validation: NFL props should have NFL-like stats
    if league and 'nfl' in league:
        nfl_stats = ['pass yards', 'passing yards', 'rush yards', 'rushing yards', 
                     'receiving yards', 'receptions', 'touchdowns', 'completions',
                     'fantasy score', 'field goals', 'kicking points']
        if not any(nfl_stat in stat_type for nfl_stat in nfl_stats):
            # This might be a misclassified prop
            return False
    
    # Basketball props should have basketball stats
    if league and ('nba' in league or 'basketball' in league):
        basketball_stats = ['points', 'rebounds', 'assists', 'blocks', 'steals', '3pt', 'three']
        if not any(bb_stat in stat_type for bb_stat in basketball_stats):
            return False
    
    return True


@st.cache_data(show_spinner=False)
def _load_grouped_cached(csv_path: str, mtime: float) -> Dict[str, List[dict]]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {k: [] for k in [
            'basketball', 'football', 'tennis', 'baseball', 'hockey', 'soccer', 'college_football',
            'csgo', 'league_of_legends', 'dota2', 'valorant', 'overwatch', 'rocket_league']}

    props: List[dict] = []
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get('name') or cols.get('player')
    line_col = cols.get('points') or cols.get('line')
    prop_col = cols.get('prop')
    sport_col = cols.get('sport')
    league_col = cols.get('league')
    team_col = cols.get('team')
    game_col = cols.get('game')
    matchup_col = cols.get('matchup')
    home_team_col = cols.get('home_team')
    away_team_col = cols.get('away_team')
    game_date_col = cols.get('game_date')
    game_time_col = cols.get('game_time')
    last_updated_col = cols.get('last_updated')
    allow_under_col = cols.get('allow_under')

    for _, row in df.iterrows():
        player_name = row.get(name_col, '') if name_col else ''
        stat_type = str(row.get(prop_col, '')).strip() if prop_col else ''
        line_val = row.get(line_col, '') if line_col else ''
        try:
            line = float(line_val) if line_val != '' else 0.0
        except Exception:
            try:
                line = float(str(line_val).split()[0])
            except Exception:
                line = 0.0
        league_val = str(row.get(league_col, '')).strip() if league_col else ''
        game_val = str(row.get(game_col, '')).strip() if game_col else ''
        sport_raw = str(row.get(sport_col, '')).strip() if sport_col else ''
        norm = _normalize_league_to_sport(league_val or game_val or sport_raw)
        sport_val = norm.lower()
        
        # Extract matchup information from enhanced CSV
        matchup_val = str(row.get(matchup_col, '')).strip() if matchup_col else ''
        home_team_val = str(row.get(home_team_col, '')).strip() if home_team_col else ''
        away_team_val = str(row.get(away_team_col, '')).strip() if away_team_col else ''

    # Compute event_start_time ISO if Game_Date/Game_Time present
        event_start_iso = ''
        try:
            gd = (str(row.get(game_date_col)) if game_date_col else '').strip()
            gt = (str(row.get(game_time_col)) if game_time_col else '').strip()
            # Expected Game_Time like "8:00 PM ET"; strip trailing timezone label
            if gd and gt:
                gt_clean = gt.replace('ET', '').strip()
                # Handle cases like "8:00 PM" or "20:00"
                dt_local = None
                try:
                    # Prefer 12-hour format
                    dt_local = datetime.strptime(f"{gd} {gt_clean}", "%Y-%m-%d %I:%M %p")
                except Exception:
                    try:
                        dt_local = datetime.strptime(f"{gd} {gt_clean}", "%Y-%m-%d %H:%M")
                    except Exception:
                        dt_local = None
                if dt_local is not None:
                    # Treat provided time as America/New_York unless specified otherwise
                    try:
                        from zoneinfo import ZoneInfo  # py3.9+
                        dt_et = dt_local.replace(tzinfo=ZoneInfo('America/New_York'))
                    except Exception:
                        dt_et = dt_local
                    event_start_iso = dt_et.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
        except Exception:
            event_start_iso = ''

        prop = {
            'player_name': player_name,
            'team': (row.get(team_col, '') if team_col else ''),
            'pick': stat_type,
            'stat_type': stat_type.lower(),
            'line': line,
            'odds': -110,
            'confidence': 50.0,
            'expected_value': 0.0,
            'avg_l10': row.get('Avg_L10') or row.get('L10') or 0.0,
            'l5_average': row.get('L5') or 0.0,
            'h2h': row.get('H2H') or row.get('h2h') or '',
            'start_time': '',
            'event_start_time': event_start_iso,  # ISO UTC when available
            'event_date': (str(row.get(game_date_col)).strip() if game_date_col else ''),
            'event_time_et': (str(row.get(game_time_col)).strip() if game_time_col else ''),
            'sport': sport_val or '',
            'league': (league_val or game_val or sport_raw).strip(),
            'over_under': None,
            'matchup': matchup_val,
            'home_team': home_team_val,
            'away_team': away_team_val,
            'last_updated': (str(row.get(last_updated_col)).strip() if last_updated_col else '')
        }
        
        # Validate prop consistency before adding
        if _validate_prop_consistency(prop):
            allow_under_val = True
            try:
                raw = row.get(allow_under_col) if allow_under_col else ''
                if isinstance(raw, str):
                    allow_under_val = str(raw).strip().lower() not in ('false', '0', 'no')
                elif isinstance(raw, (int, float)):
                    allow_under_val = bool(raw)
                elif isinstance(raw, bool):
                    allow_under_val = raw
            except Exception:
                allow_under_val = True

            # Attach flag and append
            prop['allow_under'] = allow_under_val
            props.append(prop)

    grouped = {k: [] for k in [
        'basketball', 'football', 'tennis', 'baseball', 'hockey', 'soccer', 'college_football',
        'csgo', 'league_of_legends', 'dota2', 'valorant', 'overwatch', 'rocket_league', 'apex']}

    for p in props:
        key = _map_to_sport(p)
        if key and key in grouped:
            # Double-check: ensure the prop actually belongs to this sport category
            if _validate_sport_category_match(p, key):
                grouped[key].append(p)

    return grouped

def _validate_sport_category_match(prop: dict, assigned_sport: str) -> bool:
    """
    Final validation to ensure a prop truly belongs to the assigned sport category
    """
    stat_type = str(prop.get('stat_type', '')).lower()
    league = str(prop.get('league', '')).lower()
    
    # NFL/Football validation
    if assigned_sport == 'football':
        football_stats = ['pass yards', 'passing yards', 'rush yards', 'rushing yards', 
                         'receiving yards', 'receptions', 'touchdowns', 'completions',
                         'fantasy score', 'field goals', 'kicking points', 'attempts']
        football_leagues = ['nfl', 'nfl1h', 'nfl1q', 'nfl2h']
        
        # Must have football stat OR be in football league
        has_football_stat = any(fs in stat_type for fs in football_stats)
        has_football_league = any(fl in league for fl in football_leagues)
        
        if not (has_football_stat or has_football_league):
            return False
    
    # Basketball validation
    elif assigned_sport == 'basketball':
        basketball_stats = ['points', 'rebounds', 'assists', 'blocks', 'steals', '3pt', 'three']
        basketball_leagues = ['nba', 'nbaszn', 'nbap', 'wnba', 'cbb']
        
        has_basketball_stat = any(bs in stat_type for bs in basketball_stats)
        has_basketball_league = any(bl in league for bl in basketball_leagues)
        
        if not (has_basketball_stat or has_basketball_league):
            return False
    
    # Baseball validation
    elif assigned_sport == 'baseball':
        baseball_stats = ['hits', 'home runs', 'rbis', 'strikeouts', 'total bases', 'stolen bases']
        baseball_leagues = ['mlb', 'mlblive']
        
        has_baseball_stat = any(bs in stat_type for bs in baseball_stats)
        has_baseball_league = any(bl in league for bl in baseball_leagues)
        
        if not (has_baseball_stat or has_baseball_league):
            return False
    
    # Hockey validation
    elif assigned_sport == 'hockey':
        # Require strict NHL filter — blocks any non-NHL teams/players/stats
        return hockey_strict_filter(prop)
    
    # Soccer validation  
    elif assigned_sport == 'soccer':
        soccer_leagues = ['soccer']
        if 'soccer' not in league:
            return False
    
    # CS:GO/esports validation
    elif assigned_sport == 'csgo':
        csgo_leagues = ['cs2', 'csgo', 'cs:go', 'counter-strike', 'counter strike']
        csgo_stats = ['kills', 'deaths', 'assists', 'maps', 'rounds', 'headshots', 'kd ratio', 'adr', 'rating']
        
        has_csgo_league = any(cl in league.lower() for cl in csgo_leagues)
        has_csgo_stat = any(cs in stat_type.lower() for cs in csgo_stats)
        
        if not (has_csgo_league or has_csgo_stat):
            return False
    
    # League of Legends validation
    elif assigned_sport == 'league_of_legends':
        lol_leagues = ['lol', 'league of legends', 'lcs', 'lec', 'lck', 'lpl', 'worlds', 'msi']
        lol_stats = ['kills', 'deaths', 'assists', 'cs', 'creep score', 'gold', 'damage', 'vision score', 
                    'kda', 'towers', 'dragons', 'barons', 'inhibitors', 'games', 'maps']
        
        has_lol_league = any(ll in league.lower() for ll in lol_leagues)
        has_lol_stat = any(ls in stat_type.lower() for ls in lol_stats)
        
        if not (has_lol_league or has_lol_stat):
            return False
    
    # Dota 2 validation
    elif assigned_sport == 'dota2':
        dota_leagues = ['dota2', 'dota 2', 'dota', 'ti', 'the international', 'dpc', 'major']
        dota_stats = ['kills', 'deaths', 'assists', 'last hits', 'denies', 'gpm', 'xpm', 'networth',
                     'damage', 'healing', 'tower damage', 'roshan', 'games', 'maps', 'duration']
        
        has_dota_league = any(dl in league.lower() for dl in dota_leagues)
        has_dota_stat = any(ds in stat_type.lower() for ds in dota_stats)
        
        if not (has_dota_league or has_dota_stat):
            return False
    
    # Other esports validation
    elif assigned_sport in ['valorant', 'overwatch', 'rocket_league', 'apex']:
        # For other esports, be permissive - if it's not traditional sports, allow it
        esports_leagues = ['lol', 'league of legends', 'dota2', 'dota 2', 'valorant', 'overwatch', 'rocket league', 'apex', 'apex legends']
        
        # Block if it's clearly traditional sports
        is_traditional_sport = any(indicator in f"{stat_type} {league}".lower() for indicator in [
            'nfl', 'nba', 'mlb', 'nhl', 'passing', 'rushing', 'rebounds', 'home run'
        ])
        
        if is_traditional_sport:
            return False
    
    return True


def _get_trusted_sport_data() -> Dict[str, Dict[str, List[str]]]:
    """
    Return trusted lists of teams, players, and keywords for each sport
    This is the definitive source for strict category validation
    """
    return {
        'basketball': {
            'teams': [
                'lakers', 'knicks', 'warriors', 'celtics', 'bulls', 'nets', 'suns', 'mavericks', 
                'clippers', 'spurs', 'heat', 'bucks', '76ers', 'nuggets', 'hawks', 'pelicans',
                'grizzlies', 'timberwolves', 'magic', 'wizards', 'pacers', 'pistons', 'hornets',
                'jazz', 'thunder', 'raptors', 'rockets', 'kings', 'trail blazers', 'blazers'
            ],
            'team_abbreviations': [
                'lal', 'nyk', 'gsw', 'bos', 'chi', 'bkn', 'phx', 'dal', 'lac', 'sas',
                'mia', 'mil', 'phi', 'den', 'atl', 'nop', 'mem', 'min', 'orl', 'was',
                'ind', 'det', 'cha', 'uta', 'okc', 'tor', 'hou', 'sac', 'por'
            ],
            'players': [
                'lebron james', 'stephen curry', 'kevin durant', 'giannis antetokounmpo', 
                'luka doncic', 'nikola jokic', 'joel embiid', 'jayson tatum', 'anthony davis',
                'james harden', 'damian lillard', 'jimmy butler', 'kawhi leonard', 'paul george',
                'devin booker', 'donovan mitchell', 'russell westbrook', 'chris paul',
                'victor wembanyama', 'paolo banchero', 'anthony edwards', 'ja morant',
                'trae young', 'zion williamson', 'lamelo ball', 'scottie barnes'
            ],
            'stats': ['points', 'rebounds', 'assists', 'blocks', 'steals', '3pt made', 'pts+rebs+asts', 'three pointers'],
            'leagues': ['nba', 'nbaszn', 'nbap', 'wnba', 'cbb']
        },
        'football': {
            'teams': [
                'patriots', 'saints', 'packers', 'cowboys', 'giants', 'jets', 'bears', 'steelers',
                'eagles', 'chiefs', '49ers', 'ravens', 'bengals', 'bills', 'dolphins', 'broncos',
                'lions', 'jaguars', 'panthers', 'raiders', 'chargers', 'seahawks', 'buccaneers',
                'cardinals', 'commanders', 'colts', 'vikings', 'texans', 'falcons', 'rams', 'titans'
            ],
            'team_abbreviations': [
                'ne', 'no', 'gb', 'dal', 'nyg', 'nyj', 'chi', 'pit', 'phi', 'kc', 'sf',
                'bal', 'cin', 'buf', 'mia', 'den', 'det', 'jax', 'car', 'lv', 'lac',
                'sea', 'tb', 'ari', 'was', 'ind', 'min', 'hou', 'atl', 'lar', 'ten'
            ],
            'players': [
                'josh allen', 'patrick mahomes', 'lamar jackson', 'joe burrow', 'aaron rodgers',
                'tom brady', 'dak prescott', 'russell wilson', 'kyler murray', 'derek carr',
                'tua tagovailoa', 'justin herbert', 'travis kelce', 'tyreek hill', 'davante adams',
                'cooper kupp', 'stefon diggs', 'mike evans', 'derrick henry', 'jonathan taylor',
                'christian mccaffrey', 'alvin kamara', 'dalvin cook', 'nick chubb'
            ],
            'stats': [
                'pass yards', 'passing yards', 'rush yards', 'rushing yards', 'receiving yards',
                'receptions', 'touchdowns', 'completions', 'attempts', 'fantasy score',
                'field goals', 'kicking points', 'extra points'
            ],
            'leagues': ['nfl', 'nfl1h', 'nfl1q', 'nfl2h']
        },
        'baseball': {
            'teams': [
                'yankees', 'dodgers', 'astros', 'braves', 'red sox', 'giants', 'phillies',
                'padres', 'mets', 'cardinals', 'blue jays', 'rangers', 'orioles', 'marlins',
                'cubs', 'brewers', 'twins', 'guardians', 'white sox', 'tigers', 'royals',
                'angels', 'athletics', 'mariners', 'rays', 'nationals', 'pirates', 'reds',
                'rockies', 'diamondbacks'
            ],
            'stats': ['hits', 'home runs', 'rbis', 'strikeouts', 'total bases', 'stolen bases'],
            'leagues': ['mlb', 'mlblive']
        },
        'hockey': {
            'teams': [
                'rangers', 'bruins', 'penguins', 'capitals', 'lightning', 'panthers', 'maple leafs',
                'canadiens', 'senators', 'sabres', 'devils', 'islanders', 'flyers', 'hurricanes',
                'blue jackets', 'predators', 'blues', 'wild', 'blackhawks', 'red wings',
                'avalanche', 'stars', 'jets', 'flames', 'oilers', 'canucks', 'sharks',
                'ducks', 'kings', 'golden knights', 'kraken', 'coyotes'
            ],
            'stats': ['goals', 'assists', 'shots', 'saves', 'penalty minutes', 'blocked shots'],
            'leagues': ['nhl', 'nhl1p']
        },
        'soccer': {
            'stats': ['goals', 'assists', 'shots on goal', 'shots on target', 'fouls', 'cards'],
            'leagues': ['soccer']
        },
        'tennis': {
            'stats': ['aces', 'double faults', 'games won', 'sets won'],
            'leagues': ['tennis']
        }
    }


def _strict_validate_before_render(prop: dict, target_sport: str) -> bool:
    """
    ULTRA-STRICT validation before rendering - ZERO tolerance for cross-sport contamination
    Returns True only if prop passes ALL validation checks for the target sport
    ANY doubt = REJECT
    """
    if not prop or not target_sport:
        return False
        
    player_name = str(prop.get('player_name', '')).lower()
    team = str(prop.get('team', '')).lower()
    matchup = str(prop.get('matchup', '')).lower()
    stat_type = str(prop.get('stat_type', '')).lower()
    league = str(prop.get('league', '')).lower()
    
    # ABSOLUTE BLOCKLIST - these NEVER belong in certain sports
    soccer_indicators = [
        'poland', 'lithuania', 'lewandowski', 'messi', 'ronaldo', 'mbappe', 'neymar',
        'benzema', 'haaland', 'salah', 'mane', 'de bruyne', 'modric', 'kroos',
        'real madrid', 'barcelona', 'manchester', 'liverpool', 'chelsea', 'arsenal',
        'juventus', 'bayern munich', 'psg', 'atletico', 'tottenham', 'inter',
        'ac milan', 'napoli', 'dortmund', 'ajax', 'porto', 'benfica'
    ]
    
    tennis_indicators = [
        'djokovic', 'nadal', 'federer', 'alcaraz', 'medvedev', 'tsitsipas',
        'zverev', 'rublev', 'berrettini', 'sinner', 'auger-aliassime',
        'wimbledon', 'roland garros', 'us open', 'australian open'
    ]
    
    if target_sport == 'basketball':
        # ZERO tolerance for non-basketball elements
        forbidden_elements = soccer_indicators + tennis_indicators + [
            'patriots', 'saints', 'cowboys', 'packers', 'steelers', 'eagles',
            'nfl', 'passing', 'rushing', 'receiving', 'touchdowns', 'field goal',
            'yankees', 'dodgers', 'astros', 'mlb', 'home run', 'strikeout',
            'rangers', 'bruins', 'penguins', 'nhl', 'penalty', 'goalie'
        ]
        
        # Check ALL fields for forbidden content
        all_text = f"{player_name} {team} {matchup} {stat_type} {league}"
        if any(forbidden in all_text for forbidden in forbidden_elements):
            return False
            
        # MUST have basketball league OR basketball stat
        basketball_leagues = ['nba', 'nbaszn', 'nbap', 'wnba', 'cbb']
        basketball_stats = ['points', 'rebounds', 'assists', 'blocks', 'steals', '3pt', 'three']
        
        has_basketball_league = any(bl in league for bl in basketball_leagues)
        has_basketball_stat = any(bs in stat_type for bs in basketball_stats)
        
        if not (has_basketball_league or has_basketball_stat):
            return False
            
        # MUST have recognizable NBA team OR player
        nba_teams = [
            'lakers', 'knicks', 'warriors', 'celtics', 'bulls', 'nets', 'suns', 'mavericks',
            'clippers', 'spurs', 'heat', 'bucks', '76ers', 'nuggets', 'hawks', 'pelicans',
            'grizzlies', 'timberwolves', 'magic', 'wizards', 'pacers', 'pistons', 'hornets',
            'jazz', 'thunder', 'raptors', 'rockets', 'kings', 'blazers', 'trail blazers'
        ]
        nba_abbrevs = ['lal', 'nyk', 'gsw', 'bos', 'chi', 'bkn', 'phx', 'dal', 'lac', 'sas',
                      'mia', 'mil', 'phi', 'den', 'atl', 'nop', 'mem', 'min', 'orl', 'was',
                      'ind', 'det', 'cha', 'uta', 'okc', 'tor', 'hou', 'sac', 'por']
        
        nba_players = [
            'lebron', 'curry', 'durant', 'giannis', 'luka', 'jokic', 'embiid', 'tatum',
            'davis', 'harden', 'lillard', 'butler', 'leonard', 'george', 'booker',
            'mitchell', 'westbrook', 'paul', 'wembanyama', 'banchero', 'edwards',
            'morant', 'young', 'williamson', 'ball', 'barnes'
        ]
        
        has_nba_team = any(team_name in f"{team} {matchup}" for team_name in nba_teams + nba_abbrevs)
        has_nba_player = any(player in player_name for player in nba_players)
        
        if not (has_nba_team or has_nba_player):
            return False
    
    elif target_sport == 'football':
        # ZERO tolerance for non-football elements
        forbidden_elements = soccer_indicators + tennis_indicators + [
            'lakers', 'celtics', 'warriors', 'knicks', 'bulls', 'nets',
            'nba', 'rebounds', 'assists', 'blocks', 'steals', '3pt',
            'yankees', 'dodgers', 'astros', 'mlb', 'home run', 'strikeout',
            'rangers', 'bruins', 'penguins', 'nhl', 'penalty', 'goalie'
        ]
        
        all_text = f"{player_name} {team} {matchup} {stat_type} {league}"
        if any(forbidden in all_text for forbidden in forbidden_elements):
            return False
            
        # MUST have football league OR football stat
        football_leagues = ['nfl', 'nfl1h', 'nfl1q', 'nfl2h']
        football_stats = ['pass', 'passing', 'rush', 'rushing', 'receiving', 'reception',
                         'touchdown', 'completion', 'attempt', 'fantasy', 'field goal', 'kick']
        
        has_football_league = any(fl in league for fl in football_leagues)
        has_football_stat = any(fs in stat_type for fs in football_stats)
        
        if not (has_football_league or has_football_stat):
            return False
            
        # MUST have recognizable NFL team
        nfl_teams = [
            'patriots', 'saints', 'packers', 'cowboys', 'giants', 'jets', 'bears', 'steelers',
            'eagles', 'chiefs', '49ers', 'ravens', 'bengals', 'bills', 'dolphins', 'broncos',
            'lions', 'jaguars', 'panthers', 'raiders', 'chargers', 'seahawks', 'buccaneers',
            'cardinals', 'commanders', 'colts', 'vikings', 'texans', 'falcons', 'rams', 'titans'
        ]
        nfl_abbrevs = ['ne', 'no', 'gb', 'dal', 'nyg', 'nyj', 'chi', 'pit', 'phi', 'kc', 'sf',
                      'bal', 'cin', 'buf', 'mia', 'den', 'det', 'jax', 'car', 'lv', 'lac',
                      'sea', 'tb', 'ari', 'was', 'ind', 'min', 'hou', 'atl', 'lar', 'ten']
        
        has_nfl_team = any(team_name in f"{team} {matchup}" for team_name in nfl_teams + nfl_abbrevs)
        
        if not has_nfl_team:
            return False
    
    elif target_sport == 'baseball':
        # Block non-baseball content
        forbidden_elements = soccer_indicators + tennis_indicators + [
            'patriots', 'lakers', 'nfl', 'nba', 'nhl', 'passing', 'rebounds'
        ]
        
        all_text = f"{player_name} {team} {matchup} {stat_type} {league}"
        if any(forbidden in all_text for forbidden in forbidden_elements):
            return False
            
        mlb_leagues = ['mlb', 'mlblive']
        mlb_stats = ['hits', 'home run', 'rbi', 'strikeout', 'total base', 'stolen base']
        
        has_mlb_league = any(ml in league for ml in mlb_leagues)
        has_mlb_stat = any(ms in stat_type for ms in mlb_stats)
        
        if not (has_mlb_league or has_mlb_stat):
            return False
    
    elif target_sport == 'hockey':
        # Use the centralized strict NHL filter for consistency
        return hockey_strict_filter(prop)
    
    elif target_sport == 'soccer':
        # Only allow explicit soccer content
        soccer_leagues = ['soccer']
        if 'soccer' not in league:
            return False
    
    elif target_sport == 'tennis':
        # Only allow explicit tennis content
        tennis_leagues = ['tennis']
        tennis_stats = ['aces', 'double fault', 'games won', 'sets won']
        
        has_tennis_league = 'tennis' in league
        has_tennis_stat = any(ts in stat_type for ts in tennis_stats)
        
        if not (has_tennis_league or has_tennis_stat):
            return False
    
    elif target_sport == 'csgo':
        # Allow CS:GO/CS2 esports content
        csgo_leagues = ['cs2', 'csgo', 'cs:go', 'counter-strike', 'counter strike']
        csgo_stats = ['kills', 'deaths', 'assists', 'maps', 'rounds', 'headshots', 'kd ratio', 'adr', 'rating']
        
        has_csgo_league = any(cl in league.lower() for cl in csgo_leagues) or ('esports' in league.lower())
        has_csgo_stat = any(cs in stat_type.lower() for cs in csgo_stats)
        
        if not (has_csgo_league or has_csgo_stat):
            return False
    
    elif target_sport == 'league_of_legends':
        # Allow League of Legends content
        lol_leagues = ['lol', 'league of legends', 'lcs', 'lec', 'lck', 'lpl', 'worlds', 'msi']
        lol_stats = ['kills', 'deaths', 'assists', 'cs', 'creep score', 'gold', 'damage', 'vision score', 
                    'kda', 'towers', 'dragons', 'barons', 'inhibitors', 'games', 'maps']
        
        has_lol_league = any(ll in league.lower() for ll in lol_leagues) or ('esports' in league.lower())
        has_lol_stat = any(ls in stat_type.lower() for ls in lol_stats)
        
        if not (has_lol_league or has_lol_stat):
            return False
    
    elif target_sport == 'dota2':
        # Allow Dota 2 content
        dota_leagues = ['dota2', 'dota 2', 'dota', 'ti', 'the international', 'dpc', 'major']
        dota_stats = ['kills', 'deaths', 'assists', 'last hits', 'denies', 'gpm', 'xpm', 'networth',
                     'damage', 'healing', 'tower damage', 'roshan', 'games', 'maps', 'duration']
        
        has_dota_league = any(dl in league.lower() for dl in dota_leagues) or ('esports' in league.lower())
        has_dota_stat = any(ds in stat_type.lower() for ds in dota_stats)
        
        if not (has_dota_league or has_dota_stat):
            return False
    
    elif target_sport in ['valorant', 'overwatch', 'rocket_league', 'apex']:
        # Allow other esports content - less strict validation for now
        esports_leagues = ['lol', 'league of legends', 'dota2', 'dota 2', 'valorant', 'overwatch', 'rocket league', 'apex', 'apex legends', 'esports']
        esports_stats = ['kills', 'deaths', 'assists', 'maps', 'rounds', 'damage', 'objectives', 'cs', 'gold', 'saves', 'goals', 'shots']
        
        has_esports_league = any(el in league.lower() for el in esports_leagues)
        has_esports_stat = any(es in stat_type.lower() for es in esports_stats)
        
        # For esports, be more permissive - if it doesn't match traditional sports, allow it
        is_traditional_sport = any(indicator in f"{player_name} {team} {matchup} {stat_type} {league}".lower() for indicator in [
            'nfl', 'nba', 'mlb', 'nhl', 'patriots', 'lakers', 'yankees', 'bruins',
            'passing', 'rushing', 'rebounds', 'assists', 'home run', 'penalty'
        ])
        
        if is_traditional_sport:
            return False
        
        # Allow if it has esports indicators OR doesn't match traditional sports
        return has_esports_league or has_esports_stat or not is_traditional_sport
    
    else:
        # Unknown sport - reject
        return False
    
    return True


def compute_rejection_reason(p: dict, sport: str) -> tuple[bool, str]:
    """Return (ok, reason) for strict validation results, with human-friendly reasons."""
    try:
        ok = _strict_validate_before_render(p, sport)
        if ok:
            return True, "ok"
    except Exception:
        pass

    player_name = str(p.get('player_name', '')).lower()
    team = str(p.get('team', '')).lower()
    matchup = str(p.get('matchup', '')).lower()
    stat_type = str(p.get('stat_type', '')).lower()
    league = str(p.get('league', '')).lower()
    all_text = f"{player_name} {team} {matchup} {stat_type} {league}"

    forbidden_map = {
        'basketball': ['nfl','mlb','nhl','passing','rushing','home run','pitcher','goalie'],
        'football': ['nba','mlb','nhl','rebounds','assists','3pt','goalie','home run'],
        'baseball': ['nfl','nba','nhl','passing','rebounds','goals','goalie'],
        'hockey': ['nfl','nba','mlb','passing','rebounds','home run'],
        'soccer': ['nfl','nba','mlb','nhl','rebounds','passing','home run'],
    }
    forbid = forbidden_map.get(sport, [])
    if any(x in all_text for x in forbid):
        return False, 'forbidden keyword for target sport'

    # Esports special cases
    if sport in ['csgo','league_of_legends','dota2','valorant','overwatch','rocket_league','apex']:
        if any(x in all_text for x in ['nfl','nba','mlb','nhl']):
            return False, 'looks like traditional sport'
        if (
            not any(k in league for k in ['lol','league of legends','dota','valorant','overwatch','rocket league','apex','apex legends','csgo','cs:go','cs2','counter-strike','esports'])
            and not any(k in stat_type for k in ['kill','kills','deaths','assists','maps','rounds','adr','kda','creep score','cs','gpm','xpm'])
        ):
            return False, 'no esports indicators'

    # Missing essentials
    if not player_name or not stat_type:
        return False, 'missing player/stat'

    # Stat/league mismatch generic
    if sport in ['basketball','football','baseball','hockey','soccer','tennis'] and not league:
        return False, 'missing league for traditional sport'

    return False, 'no expected stat/league for sport'


def render_validated_props_for_sport(props: List[dict], target_sport: str) -> List[dict]:
    """
    Apply strict validation before rendering props for a specific sport tab
    Log any filtered props for debugging
    """
    validated_props = []
    filtered_props: List[tuple] = []  # (prop, reason)
    
    # Check each prop and collect reasons
    show_debug_flag = False
    try:
        import streamlit as st  # type: ignore
        show_debug_flag = bool(st.session_state.get('show_debug_info', False))
    except Exception:
        show_debug_flag = False

    for prop in props:
        ok, reason = compute_rejection_reason(prop, target_sport)
        if ok:
            validated_props.append(prop)
        else:
            # In debug mode, allow soft failures to pass through for visual inspection
            if show_debug_flag and reason in ['missing player/stat', 'missing league for traditional sport', 'no expected stat/league for sport', 'no esports indicators']:
                # Mark preview so UI could distinguish later if needed
                prop['_debug_preview'] = True
                validated_props.append(prop)
            else:
                filtered_props.append((prop, reason))
    
    # Log filtered props for debugging (can be disabled in production)
    if filtered_props:
        print(f"🚫 Filtered {len(filtered_props)} props from {target_sport} tab:")
        for fp, reason in filtered_props[:5]:  # Show first 5 examples with reasons
            print(f"   - {fp.get('player_name', 'Unknown')} ({fp.get('team', 'No team')}) - {fp.get('stat_type', 'No stat')} - League: {fp.get('league', 'No league')} | Reason: {reason}")
        # Persist a sample to log for admin triage
        for fp, reason in filtered_props[:25]:
            _log_misclassified(fp, target_sport, reason)
    
    return validated_props


def load_prizepicks_csv_grouped(csv_path: str) -> Dict[str, List[dict]]:
    try:
        mtime = os.path.getmtime(csv_path)
    except Exception:
        mtime = 0.0
    return _load_grouped_cached(csv_path, mtime)


sport_emojis = {
    'basketball': '🏀', 'football': '🏈', 'tennis': '🎾', 'baseball': '⚾', 'hockey': '🏒', 'soccer': '⚽',
    'college_football': '🎓', 'csgo': '🔫', 'league_of_legends': '🧙', 'dota2': '🐉', 'valorant': '🎯', 'overwatch': '🛡️', 'rocket_league': '🚗', 'apex': '⚡'
}


def render_prop_row_html(pick: dict, sport_emoji: str) -> str:
    player_name = pick.get('player_name', 'Unknown')
    stat_label = pick.get('stat_type', '')
    line_val = pick.get('line', None)
    bet_label = pick.get('over_under', '')
    matchup = pick.get('matchup', '')
    confidence = pick.get('confidence', 0)
    edge = pick.get('ml_edge') or pick.get('expected_value', 0) / 100.0
    odds = pick.get('odds', -110)
    l5_display = f"L5 {pick.get('l5_average','-')}" if pick.get('l5_average') is not None else ""
    l10_display = f"L10 {pick.get('avg_l10','-')}" if pick.get('avg_l10') is not None else ""
    h2h_display = f"H2H {pick.get('h2h','-')}" if pick.get('h2h') is not None else ""
    
    # Add PrizePicks classification display
    prizepicks_class = pick.get('prizepicks_classification', '')
    classification_text = ''
    
    if isinstance(prizepicks_class, dict):
        # Handle dict format: {'classification': 'DISCOUNT 💰', 'emoji': '💰', ...}
        classification_text = prizepicks_class.get('classification', '')
    elif isinstance(prizepicks_class, str):
        # Handle string format: 'DISCOUNT 💰'
        classification_text = prizepicks_class
    else:
        # Handle any other format by converting to string
        classification_text = str(prizepicks_class) if prizepicks_class else ''
    
    # Clean up any potential HTML encoding issues
    classification_text = classification_text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    
    # Fallback if no classification found - generate one based on confidence
    if not classification_text:
        conf = pick.get('confidence', 0)
        if conf >= 85:
            classification_text = 'DEMON 👹'
        elif conf >= 75:
            classification_text = 'DISCOUNT 💰'
        elif conf >= 65:
            classification_text = 'DECENT ✅'
        else:
            classification_text = 'GOBLIN 👺'
    
    # Use plain text badges instead of HTML to avoid escaping issues
    type_display = f" [{classification_text}]" if classification_text else ""
    confidence_text = f"[Conf {confidence:.0f}%]"
    ev_text = f"[EV +{edge*100:.1f}%]" if edge and edge > 0 else ""
    odds_text = f"[-{abs(int(odds))}]" if isinstance(odds, (int, float)) else ""

    # Datetime display: prefer event_start_time ISO; fall back to ET string
    when_text = ''
    try:
        tz_env = os.environ.get('USER_TIMEZONE') or os.environ.get('TZ') or 'America/Chicago'
        from zoneinfo import ZoneInfo  # py3.9+
        user_tz = ZoneInfo(tz_env)
        iso = pick.get('event_start_time') or ''
        if isinstance(iso, str) and iso:
            # Normalize Z to +00:00 for fromisoformat
            dt = datetime.fromisoformat(iso.replace('Z', '+00:00'))
            dt_local = dt.astimezone(user_tz)
            when_text = dt_local.strftime('%b %d, %I:%M %p %Z')
        else:
            # Fallback: combine date/time et
            d = pick.get('event_date') or ''
            t = pick.get('event_time_et') or ''
            if d and t:
                when_text = f"{d} {t}"
    except Exception:
        when_text = pick.get('event_time_et') or ''

    # Last updated display
    updated_text = ''
    try:
        lu = pick.get('last_updated') or ''
        if lu:
            dt = datetime.fromisoformat(str(lu).replace('Z', '+00:00'))
            tz_env = os.environ.get('USER_TIMEZONE') or os.environ.get('TZ') or 'America/Chicago'
            from zoneinfo import ZoneInfo
            updated_text = dt.astimezone(ZoneInfo(tz_env)).strftime('Updated: %I:%M %p %Z')
    except Exception:
        pass

    reasoning_html = ''
    if st.session_state.get('show_reasoning') and pick.get('detailed_reasoning'):
        dr = pick['detailed_reasoning']
        summary = dr.get('summary', '')
        details = dr.get('details', [])
        lines = []
        for d in details:
            label = d.get('label', '•')
            score = d.get('score')
            reason = d.get('reason', '')
            lines.append(f"<div>• <strong>{label}</strong> {f'({score:.1f}/10)' if isinstance(score,(int,float)) else ''} – {reason}</div>")
        details_html = f"""
        <details style='margin-left:24px;'>
            <summary style='color:#8ab4f8;font-size:10px;cursor:pointer;'>Show details</summary>
            <div style='color:#9aa0a6;font-size:10px;margin-top:4px;'>
                {''.join(lines)}
            </div>
        </details>
        """ if lines else ''
        reasoning_html = f"""
        <div style='color:#9aa0a6;font-size:10px;margin-left:20px;padding:3px 0 6px;border-bottom:1px solid #222;'>
            💡 {summary}
        </div>
        {details_html}
        """

    return f"""
    <div class='prop-row' style='display:flex;align-items:center;padding:4px 0;border-bottom:1px solid #222;font-size:11px;'>
        <span style='width:20px;text-align:center;font-size:14px;'>{sport_emoji}</span>
        <span style='flex:1.5;font-weight:600;color:#fff;font-size:12px;'>{player_name}{type_display}</span>
        <span style='flex:1.2;color:#e8e8e8;font-size:11px;'>{bet_label or ''} {line_val if line_val is not None else ''} {stat_label}</span>
        <span style='flex:1;color:#b8b8b8;font-size:10px;'>{matchup}</span>
        {f"<span style='flex:1;color:#9aa0a6;font-size:10px;'>{when_text}</span>" if when_text else ''}
        {f"<span style='flex:0.6;color:#9aa0a6;font-size:10px;'>{l5_display}</span>" if l5_display else ''}
        {f"<span style='flex:0.6;color:#9aa0a6;font-size:10px;'>{l10_display}</span>" if l10_display else ''}
        {f"<span style='flex:0.6;color:#9aa0a6;font-size:10px;'>{h2h_display}</span>" if h2h_display else ''}
        <span style='min-width:80px;text-align:right;color:#34a853;font-size:10px;'>{confidence_text}</span>
        <span style='min-width:80px;text-align:right;color:#0f9d58;font-size:10px;'>{ev_text}</span>
        <span style='min-width:60px;text-align:right;color:#1a73e8;font-size:10px;'>{odds_text}</span>
        {f"<span class='pill' style='margin-left:8px;color:#9aa0a6;'>{updated_text}</span>" if updated_text else ''}
    </div>
    {reasoning_html}
    """


def display_sport_page(sport_key: str, title: str, AgentClass, cap: int = 200) -> None:
    st.set_page_config(page_title=f"{title} - BetFinder AI", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

    st.markdown("""
    <style>
      .pill { border-radius: 999px; padding: 3px 10px; font-size: 10px; margin-left: 4px; }
      .pill-edge { background: #0f9d58; color: #fff; }
      .pill-odds { background: #1a73e8; color: #fff; }
      .badge-high { background: #34a853; color: #111; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<h2>{title} {sport_emojis.get(sport_key,'')}</h2>", unsafe_allow_html=True)
    # Fallback auto-refresh using meta tag (60s)
    st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)

    # Auto-refresh every minute to pick up new scrapes
    # If a native autorefresh util is available, use it (optional)
    try:
        if hasattr(st, 'autorefresh'):
            st.autorefresh(interval=60_000, key=f"autorefresh_{sport_key}")
    except Exception:
        pass

    # Controls
    st.sidebar.header("Live Data")
    csv_path = get_effective_csv_path()
    st.sidebar.write(f"CSV path: `{csv_path}`")
    # Timezone picker (optional)
    try:
        default_tz = os.environ.get('USER_TIMEZONE') or os.environ.get('TZ') or 'America/Chicago'
        st.session_state['user_timezone'] = st.sidebar.text_input('Timezone', value=default_tz, help='IANA TZ, e.g., America/Chicago')
        os.environ['USER_TIMEZONE'] = st.session_state['user_timezone']
    except Exception:
        pass
    
    # Cache controls
    if st.sidebar.button("🔄 Force Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    if st.sidebar.button("🧹 Clear All Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    # Debug controls (default ON to surface triage info)
    show_debug = st.sidebar.checkbox("🔍 Show Debug Info", True)
    try:
        st.session_state['show_debug_info'] = bool(show_debug)
    except Exception:
        pass

    # Ensure freshness
    interval = int(os.environ.get('AUTO_SCRAPE_INTERVAL_SEC', '60'))
    ensure_fresh_csv(csv_path, interval, target_sport=sport_key)

    # Load grouped props
    grouped = load_prizepicks_csv_grouped(csv_path)
    items = grouped.get(sport_key, [])

    # Apply STRICT validation before rendering
    validated_items = render_validated_props_for_sport(items, sport_key)

    # Filter out expired/started events (keep only future or now)
    def _is_future(prop: dict) -> bool:
        iso = prop.get('event_start_time') or ''
        if not iso:
            # If no time info, keep (can't determine)
            return True
        try:
            dt = datetime.fromisoformat(iso.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            return dt >= now
        except Exception:
            return True
    validated_items = [p for p in validated_items if _is_future(p)]

    # NHL-specific safety message if any invalid props were detected
    if sport_key == 'hockey' and items and len(validated_items) < len(items):
        st.warning("Invalid prop detected—see admin/log for details.")

    # Show debug info if enabled
    if show_debug:
        st.sidebar.markdown("### Debug Info")
        st.sidebar.write(f"Raw props: {len(items)}")
        st.sidebar.write(f"Validated props: {len(validated_items)}")
        st.sidebar.write(f"Filtered out: {len(items) - len(validated_items)}")
        
        if len(items) != len(validated_items):
            with st.sidebar.expander("🚫 Filtered Props"):
                # Recompute with reasons for display
                reasons = []
                for p in items:
                    if p not in validated_items:
                        ok, r = compute_rejection_reason(p, sport_key)
                        if not ok:
                            reasons.append((p, r))
                for i, (fp, r) in enumerate(reasons[:10]):
                    st.write(f"{i+1}. {fp.get('player_name', 'Unknown')} - {fp.get('stat_type', 'Unknown')} - League: {fp.get('league', 'Unknown')} — Reason: {r}")

    if not validated_items:
        if items:
            # Some props existed but all were rejected by safety filters
            st.error("🚫 Invalid prop detected—see admin/log for details.")
            st.warning(f"All {len(items)} props for {title} failed strict validation checks to prevent cross-sport contamination.")
            
            # Show admin info if debug mode is on
            if show_debug:
                with st.expander("🔍 Admin Debug - Why props were rejected"):
                    for i, item in enumerate(items[:3]):
                        st.text(f"{i+1}. {item.get('player_name', 'Unknown')} - {item.get('stat_type', 'Unknown')} - League: {item.get('league', 'Unknown')} - Team: {item.get('team', 'Unknown')}")
        else:
            # No props pulled for this sport at all — do not use mock/merge
            if sport_key == 'hockey':
                st.info("ℹ️ Live Hockey props not available. Please try again later.")
            else:
                st.info(f"ℹ️ **No {title} props available right now**")
                st.write("Please check back later for new prop data.")
        return

    # Show filtering stats if any props were removed
    if len(items) != len(validated_items):
        st.info(f"📊 Showing {len(validated_items)} validated props (filtered out {len(items) - len(validated_items)} inconsistent props)")

    # Build picks using the agent, without logging
    agent = AgentClass()
    capped = validated_items[:cap]
    picks = agent.make_picks(props_data=capped, log_to_ledger=False)

    # Header: show last updated for transparency
    # Try to compute most recent Last_Updated across picks, fallback to CSV mtime
    last_updated_display = ''
    try:
        tz_env = os.environ.get('USER_TIMEZONE') or os.environ.get('TZ') or 'America/Chicago'
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_env)
        lus = []
        for p in capped:
            lu = p.get('last_updated')
            if lu:
                try:
                    lus.append(datetime.fromisoformat(str(lu).replace('Z', '+00:00')))
                except Exception:
                    continue
        if lus:
            last_dt = max(lus).astimezone(tz)
            last_updated_display = last_dt.strftime('%-I:%M %p %Z') if hasattr(last_dt, 'strftime') else ''
        else:
            # Fallback to file mtime
            mtime = os.path.getmtime(csv_path)
            last_dt = datetime.fromtimestamp(mtime, tz)
            last_updated_display = last_dt.strftime('%-I:%M %p %Z')
    except Exception:
        pass

    # Render
    st.markdown(
        f"<div class='section-title'>{title}<span class='time' style='margin-left:8px;opacity:0.8;'>{len(picks)} shown</span>"
        + (f"<span class='pill' style='margin-left:12px;background:#222;color:#9aa0a6;'>Updated: {last_updated_display}</span>" if last_updated_display else '')
        + "</div>",
        unsafe_allow_html=True,
    )
    for p in picks:
        html_row = render_prop_row_html(p, sport_emojis.get(sport_key, ''))
        st.markdown(html_row, unsafe_allow_html=True)

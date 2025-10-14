import os
from typing import Dict, List, Any
import streamlit as st

sport_emojis = {
    'basketball': '🏀', 'football': '🏈', 'tennis': '🎾', 'baseball': '⚾', 'hockey': '🏒', 'soccer': '⚽',
    'college_football': '🎓', 'csgo': '🔫', 'league_of_legends': '🧙', 'dota2': '🐉', 'valorant': '🎯', 'overwatch': '🛡️', 'rocket_league': '🚗', 'apex': '⚡'
}


def _safe_str(val: Any) -> str:
    if val is None:
        import os
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        from typing import Dict, List
        import streamlit as st

        sport_emojis = {
            'basketball': '🏀', 'football': '🏈', 'tennis': '🎾', 'baseball': '⚾', 'hockey': '🏒', 'soccer': '⚽',
            'college_football': '🎓', 'csgo': '🔫', 'league_of_legends': '🧙', 'dota2': '🐉', 'valorant': '🎯', 'overwatch': '🛡️', 'rocket_league': '🚗', 'apex': '⚡'
        }

        def render_prop_row_html(pick: dict, sport_emoji: str) -> str:
            player_name = pick.get('player_name', 'Unknown')
            stat_label = pick.get('stat_type', '')
            line_val = pick.get('line', None)
            bet_label = pick.get('over_under', '')
            matchup = pick.get('matchup', 'N/A')
            confidence = pick.get('confidence', 0)
            # keep a small guard for missing values
            player_name = player_name or 'Unknown'
            # Note: function will gracefully handle missing keys

            return f"""
            <div class='prop-row' style='display:flex;align-items:center;padding:4px 0;border-bottom:1px solid #222;font-size:11px;'>
                <span style='width:20px;text-align:center;font-size:14px;'>{sport_emoji}</span>
                <span style='flex:1.5;font-weight:600;color:#fff;font-size:12px;'>{player_name}</span>
                <span style='flex:1.2;color:#e8e8e8;font-size:11px;'>{bet_label or ''} {line_val if line_val is not None else ''} {stat_label}</span>
                <span style='flex:1;color:#b8b8b8;font-size:10px;'>{matchup}</span>
                {f"<span style='flex:1;color:#9aa0a6;font-size:10px;'>...</span>"}
            </div>
            """


        def display_sport_page(sport_key: str, title: str, AgentClass, cap: int = 200) -> None:
            st.set_page_config(page_title=f"{title} - BetFinder AI", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

            st.markdown(f"<h2>{title} {sport_emojis.get(sport_key,'')}</h2>", unsafe_allow_html=True)
    
            with st.spinner(f"Fetching live {title} projections..."):
                agent = AgentClass()
                picks = agent.make_picks(log_to_ledger=False, max_props=cap)

            if not picks:
                st.info(f"ℹ️ **No {title} props available right now.**")
                return

            st.markdown(
                f"<div class='section-title'>{title}<span class='time' style='margin-left:8px;opacity:0.8;'>{len(picks)} props found</span></div>",
                unsafe_allow_html=True,
            )

            for p in picks:
                html_row = render_prop_row_html(p, sport_emojis.get(sport_key, ''))
                st.markdown(html_row, unsafe_allow_html=True)

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

    # Esports special cases (UPDATED BLOCK FOR CONTAMINATION CHECK)
    if sport in ['csgo','league_of_legends','dota2','valorant','overwatch','rocket_league','apex']:
        traditional_contamination_indicators = [
            'nfl', 'nba', 'mlb', 'nhl', 'passing', 'rushing', 'rebounds', 'assists', 'home run', 
            'strikeout', 'field goal', 'penalty', 'goalie', 'lal', 'gsw', 'bos', 'chi', 'nyk', 'phi',
            'lebron', 'curry', 'mahomes', 'tatum', 'trout', 'crosby', 'ovechkin', 'basketball', 'football'
        ]
        if any(x in all_text for x in traditional_contamination_indicators):
            return False, 'looks like traditional sport (cross-sport contamination check)'
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
    stat_label = pick.get('stat_type', '').capitalize()
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
            lines.append(f"• <strong>{label}</strong> {f'({score:.1f}/10)' if isinstance(score,(int,float)) else ''} – {reason}")
        details_html = f"""
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
            lines.append(f"• <strong>{label}</strong> {f'({score:.1f}/10)' if isinstance(score,(int,float)) else ''} – {reason}")
        details_html = f"""
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
            lines.append(f"• <strong>{label}</strong> {f'({score:.1f}/10)' if isinstance(score,(int,float)) else ''} – {reason}")
        details_html = f"""
        <details style='margin-left:24px;'>
            <summary style='color:#8ab4f8;font-size:10px;cursor:pointer;'>Show details</summary>
            <div style='color:#9aa0a6;font-size:10px;margin-top:4px;'>
                {''.join(lines)}
            # Removed stray </div>
        </details>
        """ if lines else ''
        reasoning_html = f"""
        <div style='color:#9aa0a6;font-size:10px;margin-left:20px;padding:3px 0 6px;border-bottom:1px solid #222;'>
            💡 {summary}
    # Removed stray </div>
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
    # Removed stray </div>
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
    # Use markdown and emojis for formatting, then render picks in a 2-column layout
    section_title = f"### {title}  ({len(picks)} shown)"
    if last_updated_display:
        section_title += f"  **Updated:** {last_updated_display}"
    st.markdown(section_title)

    # --- START LAYOUT & PERFORMANCE LIMIT ---
    MAX_CARDS_TO_DISPLAY = 20
    NUM_COLUMNS = 2

    display_picks = picks[:MAX_CARDS_TO_DISPLAY]

    if not display_picks:
        st.info(f"No picks generated for {title} (0/{len(picks)} props passed filters).")
        return

    cols = st.columns(NUM_COLUMNS)
    for i, p in enumerate(display_picks):
        current_col = cols[i % NUM_COLUMNS]
        with current_col:
            html_row = render_prop_row_html(p, sport_emojis.get(sport_key, ''))
            st.markdown(html_row, unsafe_allow_html=True)

    if len(picks) > MAX_CARDS_TO_DISPLAY:
        st.info(f"Showing first {MAX_CARDS_TO_DISPLAY} picks out of {len(picks)} total for performance.")
    # --- END LAYOUT & PERFORMANCE LIMIT ---

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List

import pandas as pd
import streamlit as st
import requests


def get_effective_csv_path() -> str:
    if 'prizepicks_csv_path' not in st.session_state:
        st.session_state['prizepicks_csv_path'] = 'prizepicks_props.csv'
    return st.session_state['prizepicks_csv_path']


def ensure_fresh_csv(path: str, max_age_sec: int = 300) -> None:
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = 0
    age = time.time() - mtime if mtime else 1e9
    if age > max(30, 2 * max_age_sec):
        try:
            os.environ['PRIZEPICKS_CSV'] = path
            from prizepicks_scrape import main as scrape_main
            scrape_main()
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
        'csgo': 'csgo', 'cs:go': 'csgo', 'cs2': 'csgo', 'counter-strike': 'csgo', 'counter strike': 'csgo', 'counter-strike 2': 'csgo'
    }
    return aliases.get(s, s)


def _map_to_sport(p: dict) -> str:
    player = str(p.get('player_name', '')).lower()
    stat = str(p.get('stat_type', '')).lower()

    if p.get('sport'):
        s = str(p['sport']).lower()
        aliases = {
            'nba': 'basketball', 'wnba': 'basketball', 'cbb': 'basketball',
            'nfl': 'football', 'cfb': 'college_football',
            'mlb': 'baseball', 'nhl': 'hockey', 'epl': 'soccer',
            'lol': 'league_of_legends', 'league of legends': 'league_of_legends', 'league_of_legends': 'league_of_legends',
            'dota2': 'dota2', 'dota 2': 'dota2',
            'valorant': 'valorant', 'valo': 'valorant',
            'overwatch': 'overwatch', 'overwatch 2': 'overwatch', 'ow': 'overwatch',
            'rocket league': 'rocket_league', 'rocket_league': 'rocket_league', 'rl': 'rocket_league',
            'csgo': 'csgo', 'cs:go': 'csgo', 'cs2': 'csgo', 'counter-strike': 'csgo', 'counter strike': 'csgo', 'counter-strike 2': 'csgo'
        }
        mapped = aliases.get(s)
        if mapped:
            return mapped

    nba_players = ['stephen curry', 'kevin durant', 'lebron james', 'giannis antetokounmpo', 'james harden',
                   'luka doncic', 'nikola jokic', 'joel embiid', 'jayson tatum', 'anthony davis']
    basketball_stats = ['points', 'rebounds', 'assists', 'blocks', 'steals', '3pt made', 'pts+rebs+asts']
    if any(n in player for n in nba_players) or any(k in stat for k in basketball_stats):
        return 'basketball'

    college_players = ['caleb williams', 'drake maye', 'bo nix', 'michael penix jr', 'marvin harrison jr']
    football_stats = ['pass yards', 'passing yards', 'rush yards', 'rushing yards', 'receiving yards', 'receptions', 'touchdown']
    if any(n in player for n in college_players) and any(k in stat for k in football_stats):
        return 'college_football'
    if any(k in stat for k in football_stats):
        return 'football'

    csgo_players = ['s1mple', 'niko', 'zywoo', 'm0nesy', 'ropz', 'broky']
    dota_players = ['sumail', 'miracle', 'yatoro', 'ame']
    lol_players = ['faker', 'ruler', 'chovy', 'knight', 'caps', 'showmaker']
    valorant_players = ['tenz', 'scream', 'aspas', 'derke', 'yay']
    overwatch_players = ['profit', 'fleta', 'carpe']
    rl_players = ['apparentlyjack', 'firstkiller', 'atomic', 'jknaps']

    if any(n in player for n in dota_players):
        return 'dota2'
    if any(n in player for n in lol_players):
        return 'league_of_legends'
    if any(n in player for n in valorant_players):
        return 'valorant'
    if any(n in player for n in overwatch_players):
        return 'overwatch'
    if any(n in player for n in rl_players):
        return 'rocket_league'
    if any(n in player for n in csgo_players):
        return 'csgo'

    lol_keywords = ['kda', 'k/d/a', 'kills+assists', 'kills + assists', 'creep score', 'cs ']
    if any(k in stat for k in lol_keywords):
        return 'league_of_legends'
    dota_keywords = ['gpm', 'xpm', 'last hits', 'denies', 'roshan']
    if any(k in stat for k in dota_keywords):
        return 'dota2'
    if ('map' in stat or 'maps' in stat) and 'kills' in stat:
        if any(n in player for n in csgo_players):
            return 'csgo'
        if any(n in player for n in valorant_players):
            return 'valorant'
        if any(n in player for n in dota_players):
            return 'dota2'
        return ''
    valorant_keywords = ['acs', 'first bloods', 'first kills', 'spike']
    if any(k in stat for k in valorant_keywords):
        return 'valorant'
    overwatch_keywords = ['eliminations', 'final blows', 'objective', 'healing']
    if any(k in stat for k in overwatch_keywords):
        return 'overwatch'
    rocket_league_keywords = ['rocket league', 'rl goals', 'rl assists', 'rl saves', 'rl shots']
    if any(k in stat for k in rocket_league_keywords):
        return 'rocket_league'
    csgo_keywords = ['headshot', 'headshots', 'awp', 'adr', 'clutch', 'clutches']
    if any(k in stat for k in csgo_keywords):
        return 'csgo'

    nhl_players = ['connor mcdavid', 'leon draisaitl', 'sidney crosby', 'auston matthews']
    if any(n in player for n in nhl_players):
        return 'hockey'
    hockey_specific = ['penalty minutes', 'power play', 'faceoff', 'time on ice', 'plus/minus', 'blocked shots', 'goalie saves']
    if any(k in stat for k in hockey_specific):
        return 'hockey'
    if 'saves' in stat and any(k in stat for k in ['goalie', 'save percentage', 'goals against']):
        return 'hockey'

    soccer_stats = ['goals', 'assists', 'shots on goal', 'shots on target', 'shots', 'goal + assist', 'fouls', 'cards', 'clean sheets', 'saves', 'goalie saves']
    if any(k in stat for k in soccer_stats):
        return 'soccer'

    if any(k in stat for k in ['strokes', 'birdies', 'eagles', 'pars', 'bogeys']):
        return ''
    if any(k in stat for k in ['aces', 'double faults', 'games won', 'sets won']):
        return 'tennis'
    if any(k in stat for k in ['hits', 'home runs', 'rbis', 'strikeouts', 'total bases', 'stolen bases']):
        return 'baseball'

    return ''


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
    game_col = cols.get('game')

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

        prop = {
            'player_name': player_name,
            'team': '',
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
            'sport': sport_val or '',
            'league': (league_val or game_val or sport_raw).strip(),
            'over_under': None
        }
        props.append(prop)

    grouped = {k: [] for k in [
        'basketball', 'football', 'tennis', 'baseball', 'hockey', 'soccer', 'college_football',
        'csgo', 'league_of_legends', 'dota2', 'valorant', 'overwatch', 'rocket_league']}

    for p in props:
        key = _map_to_sport(p)
        if key and key in grouped:
            grouped[key].append(p)

    return grouped


def load_prizepicks_csv_grouped(csv_path: str) -> Dict[str, List[dict]]:
    try:
        mtime = os.path.getmtime(csv_path)
    except Exception:
        mtime = 0.0
    return _load_grouped_cached(csv_path, mtime)


sport_emojis = {
    'basketball': '🏀', 'football': '🏈', 'tennis': '🎾', 'baseball': '⚾', 'hockey': '🏒', 'soccer': '⚽',
    'college_football': '🎓', 'csgo': '🔫', 'league_of_legends': '🧙', 'dota2': '🐉', 'valorant': '🎯', 'overwatch': '🛡️', 'rocket_league': '🚗'
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
    l5_display = f"L5 {pick.get('l5_average','-')}"
    l10_display = f"L10 {pick.get('avg_l10','-')}"
    h2h_display = f"H2H {pick.get('h2h','-')}"

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
        <span style='flex:1.5;font-weight:600;color:#fff;font-size:12px;'>{player_name}</span>
        <span style='flex:1.2;color:#e8e8e8;font-size:11px;'>{bet_label or ''} {line_val if line_val is not None else ''} {stat_label}</span>
        <span style='flex:1;color:#b8b8b8;font-size:10px;'>{matchup}</span>
        <span style='flex:0.6;color:#9aa0a6;font-size:10px;'>{l5_display}</span>
        <span style='flex:0.6;color:#9aa0a6;font-size:10px;'>{l10_display}</span>
        <span style='flex:0.6;color:#9aa0a6;font-size:10px;'>{h2h_display}</span>
        <span style='min-width:50px;text-align:right;'><span class='pill badge-high' style='font-size:9px;padding:2px 6px;'>Conf {confidence:.0f}%</span></span>
        <span style='min-width:50px;text-align:right;'>{f"<span class='pill pill-edge' style='font-size:9px;padding:2px 6px;'>EV +{edge*100:.1f}%</span>" if edge and edge > 0 else ''}</span>
        <span style='min-width:40px;text-align:right;'>{f"<span class='pill pill-odds' style='font-size:9px;padding:2px 6px;'>-{abs(int(odds))}</span>" if isinstance(odds, (int, float)) else ''}</span>
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

    # Controls
    st.sidebar.header("Live Data")
    csv_path = get_effective_csv_path()
    st.sidebar.write(f"CSV path: `{csv_path}`")

    # Ensure freshness
    interval = int(os.environ.get('AUTO_SCRAPE_INTERVAL_SEC', '60'))
    ensure_fresh_csv(csv_path, interval)

    # Load grouped props
    grouped = load_prizepicks_csv_grouped(csv_path)
    items = grouped.get(sport_key, [])

    if not items:
        st.warning(f"No prop lines available for {title}.")
        return

    # Build picks using the agent, without logging
    from sport_agents import SportAgent  # type: ignore
    agent = AgentClass()
    capped = items[:cap]
    picks = agent.make_picks(props_data=capped, log_to_ledger=False)

    # Render
    st.markdown(
        f"<div class='section-title'>{title}<span class='time' style='margin-left:8px;opacity:0.8;'>{len(picks)} shown</span></div>",
        unsafe_allow_html=True,
    )
    rows = []
    for p in picks:
        rows.append(render_prop_row_html(p, sport_emojis.get(sport_key, '')))
    st.markdown("\n".join(rows), unsafe_allow_html=True)

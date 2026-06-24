import pandas as pd
from DrissionPage import ChromiumPage, ChromiumOptions
import time
import os
import re
import sys
import random


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ============================================================
# LEAGUE CONFIGURATION
# ============================================================
LEAGUES = {
    "premier_league": {
        "name": "Premier League",
        "flag": "PL",
        "comp_id": "9",
        "league_name": "Premier-League",
        "table_id": "stats_standard",
        "output_raw": "t_raw.csv",
        "output_photos": "t_with_photos.csv",
        "output_clean": "t_cleaned.csv",
    },
    "ligue1": {
        "name": "Ligue 1",
        "flag": "L1",
        "comp_id": "13",
        "league_name": "Ligue-1",
        "table_id": "stats_standard",
        "output_raw": "l_raw.csv",
        "output_photos": "l_photos.csv",
        "output_clean": "l_cleaned.csv",
    },
    "laliga": {
        "name": "La Liga",
        "flag": "LL",
        "comp_id": "12",
        "league_name": "La-Liga",
        "table_id": "stats_standard",
        "output_raw": "laliga_players_raw.csv",
        "output_photos": "laliga_players_with_photos.csv",
        "output_clean": "laliga_players_cleaned.csv",
    },
    "bundesliga": {
        "name": "Bundesliga",
        "flag": "BL",
        "comp_id": "20",
        "league_name": "Bundesliga",
        "table_id": "stats_standard",
        "output_raw": "bundesliga_players_raw.csv",
        "output_photos": "bundesliga_players_with_photos.csv",
        "output_clean": "bundesliga_players_cleaned.csv",
    },
    "seriea": {
        "name": "Serie A",
        "flag": "SA",
        "comp_id": "11",
        "league_name": "Serie-A",
        "table_id": "stats_standard",
        "output_raw": "seriea_players_raw.csv",
        "output_photos": "seriea_players_with_photos.csv",
        "output_clean": "seriea_players_cleaned.csv",
    },
}

# ============================================================
# ANTI-BLOCK HELPERS
# ============================================================
def human_delay(min_s=3.0, max_s=7.0):
    time.sleep(random.uniform(min_s, max_s))

def make_page(port=9222):
    co = ChromiumOptions()
    co.set_local_port(port)
    co.set_user_data_path(f"/tmp/drission_fbref_{port}")
    return ChromiumPage(co)

def safe_get(page, url):
    """Navigate to an absolute URL — use JS to bypass DrissionPage relative-URL bug."""
    url = str(url).strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    try:
        page.run_js(f'window.location.href = "{url}";')
    except Exception:
        page.get(url)


# ============================================================
# STEP 1 - Scrape FBref stats table
# ============================================================
def scrape_stats(league_key, season=None, output_dir=".", log_fn=None):
    def log(msg):
        print(msg, flush=True)
        if log_fn:
            log_fn(msg)

    league = LEAGUES[league_key]
    log(f"{'='*60}")
    log(f"  Scraping stats: {league['name']} (Saison: {season if season and season != 'Actuelle' else 'Actuelle'})")
    log(f"{'='*60}")

    raw_path = os.path.join(output_dir, league["output_raw"])

    if season and season != "Actuelle":
        url = f"https://fbref.com/en/comps/{league['comp_id']}/{season}/stats/{season}-{league['league_name']}-Stats"
    else:
        url = f"https://fbref.com/en/comps/{league['comp_id']}/stats/{league['league_name']}-Stats"

    page = make_page(port=9222)
    try:
        log(f"Opening: {url}")
        safe_get(page, url)
        human_delay(4.0, 8.0)

        table_el = page.ele(f"#{league['table_id']}", timeout=5)
        if not table_el:
            table_el = page.ele("t:table@@id:stats_standard", timeout=5)

        if not table_el:
            log("Stats table not found. Try again or check the FBref URL.")
            return None

        headers = []
        thead = table_el.ele("t:thead", timeout=2)
        if thead:
            all_header_rows = thead.eles("t:tr")
            last_row = all_header_rows[-1]
            for th in last_row.eles("t:th"):
                stat = th.attr("data-stat") or th.text.strip()
                headers.append(stat)
        headers.append("Player_URL")

        rows = []
        tbody = table_el.ele("t:tbody", timeout=2)
        if tbody:
            for tr in tbody.eles("t:tr"):
                cls = tr.attr("class") or ""
                if "thead" in cls:
                    continue
                cells = tr.eles("xpath:.//th | .//td")
                row = []
                player_url = ""
                for cell in cells:
                    a = cell.ele("t:a", timeout=0)
                    if a:
                        href = a.attr("href") or ""
                        if "/players/" in href and not player_url:
                            if href.startswith("http"):
                                player_url = href
                            else:
                                player_url = "https://fbref.com" + href
                    row.append(cell.text.strip() if cell.text else "")
                if len(row) == len(headers) - 1:
                    row.append(player_url)
                    rows.append(row)

        if not rows:
            log("No rows found in the table.")
            return None

        df = pd.DataFrame(rows, columns=headers)
        df.to_csv(raw_path, index=False, encoding="utf-8-sig")
        log(f"Raw stats saved: {raw_path} ({len(df)} rows)")
        return raw_path

    finally:
        page.quit()


# ============================================================
# STEP 2 - Scrape player photos
# ============================================================
def extract_photo_from_page(page):
    meta_div = page.ele("#meta", timeout=3)
    if meta_div:
        img = meta_div.ele("t:img", timeout=2)
        if img:
            src = img.attr("src") or ""
            if "headshots" in src and "silhouette" not in src:
                return src
    return None


def scrape_photos(league_key, output_dir=".", log_fn=None):
    def log(msg):
        print(msg, flush=True)
        if log_fn:
            log_fn(msg)

    league = LEAGUES[league_key]
    raw_path = os.path.join(output_dir, league["output_raw"])
    photos_path = os.path.join(output_dir, league["output_photos"])
    progress_path = os.path.join(output_dir, "progress_" + league["output_photos"])

    log(f"{'='*60}")
    log(f"  Scraping photos: {league['name']}")
    log(f"{'='*60}")

    if not os.path.exists(raw_path):
        log(f"Raw file not found: {raw_path}. Run scrape_stats() first.")
        return None

    df = pd.read_csv(raw_path)
    url_cols = [c for c in df.columns if "Player_URL" in str(c)]
    if not url_cols:
        log("Column 'Player_URL' not found in raw CSV.")
        return None
    url_col = url_cols[0]

    results = [""] * len(df)
    start_index = 0
    if os.path.exists(progress_path):
        prog_df = pd.read_csv(progress_path)
        if "Image_URL" in prog_df.columns:
            saved = list(prog_df["Image_URL"].fillna(""))
            start_index = len(saved)
            for i, v in enumerate(saved):
                results[i] = v
            log(f"Resuming from player {start_index + 1}...")

    name_cols = [c for c in df.columns if "player" in str(c).lower() and "url" not in str(c).lower()]

    NUM_THREADS = 5
    BASE_PORT = 9340  # each thread gets its own port: 9340..9344

    import threading
    import queue

    q = queue.Queue()
    for i in range(start_index, len(df)):
        q.put(i)

    results_lock = threading.Lock()
    progress_lock = threading.Lock()
    completed = [0]

    log(f"Lancement de {NUM_THREADS} navigateurs DrissionPage en parallele...")

    def worker(thread_id):
        port = BASE_PORT + thread_id
        page = make_page(port=port)
        try:
            while True:
                try:
                    i = q.get_nowait()
                except queue.Empty:
                    break

                row = df.iloc[i]
                url = str(row[url_col]).strip()
                name = row[name_cols[0]] if name_cols else f"Player {i}"

                photo_url = ""
                if url and url not in ("nan", "Player_URL") and url.startswith("https://"):
                    log(f"[T{thread_id}][{i+1}/{len(df)}] {name}...")
                    try:
                        safe_get(page, url)
                        human_delay(2.5, 5.5)
                        photo_url = extract_photo_from_page(page)
                        if not photo_url:
                            human_delay(2.0, 4.0)
                            photo_url = extract_photo_from_page(page)
                        log(f"  [T{thread_id}] -> {'OK' if photo_url else 'Not found'}")
                    except Exception as e:
                        log(f"  [T{thread_id}] -> Error: {str(e)[:60]}")

                with results_lock:
                    results[i] = photo_url if photo_url else ""

                with progress_lock:
                    completed[0] += 1
                    if completed[0] % 10 == 0:
                        temp_df = df.copy()
                        temp_df["Image_URL"] = results
                        temp_df.to_csv(progress_path, index=False)

                q.task_done()
        finally:
            page.quit()
            log(f"  [T{thread_id}] Navigateur ferme.")

    threads = []
    for t_id in range(NUM_THREADS):
        t = threading.Thread(target=worker, args=(t_id,), daemon=True)
        threads.append(t)
        t.start()
        time.sleep(2)

    for t in threads:
        t.join()

    df["Image_URL"] = results
    if os.path.exists(progress_path):
        os.remove(progress_path)

    df.to_csv(photos_path, index=False, encoding="utf-8-sig")
    found = sum(1 for u in results[start_index:] if u)
    log(f"Done! Photos found: {found}/{len(df)}")
    log(f"Saved: {photos_path}")
    return photos_path


# ============================================================
# STEP 3 - Clean the CSV
# ============================================================
def clean_csv(league_key, output_dir=".", log_fn=None):
    def log(msg):
        print(msg, flush=True)
        if log_fn:
            log_fn(msg)

    league = LEAGUES[league_key]
    photos_path = os.path.join(output_dir, league["output_photos"])
    clean_path = os.path.join(output_dir, league["output_clean"])

    if not os.path.exists(photos_path):
        log(f"Photos file not found: {photos_path}")
        return None

    log(f"--- Cleaning {photos_path} ---")
    
    # Load the CSV
    df = pd.read_csv(photos_path)
    
    # 1. Clean Column Names
    new_columns = []
    for col in df.columns:
        clean_name = re.sub(r'^Unnamed: \d+_level_0_', '', col)
        new_columns.append(clean_name)
    df.columns = new_columns
    
    # 2. Remove intermediate header rows
    if 'Player' in df.columns:
        df = df[df['Player'] != 'Player']
    if 'Rk' in df.columns:
        df = df[df['Rk'] != 'Rk']
    
    # 3. Handle 'Age' column
    if 'Age' in df.columns:
        def convert_age(age_str):
            if pd.isna(age_str) or not isinstance(age_str, str):
                return age_str
            parts = age_str.split('-')
            if len(parts) == 2:
                try:
                    years = float(parts[0])
                    days = float(parts[1])
                    return round(years + (days / 365.25), 2)
                except ValueError:
                    return age_str
            return age_str
        df['Age'] = df['Age'].apply(convert_age)

    # 4. Clean 'Nation'
    if 'Nation' in df.columns:
        df['Nation'] = df['Nation'].str.split(' ').str[-1]

    # 5. Drop irrelevant columns
    cols_to_drop = [
        'Matches', 'Player_URL', 'Matches_URL', 'Rk', 
        'Unnamed: 24_level_0_Matches'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # 6. Convert numeric columns to float/int
    categorical_cols = ['Player', 'Nation', 'Pos', 'Squad', 'Born', 'Image_URL']
    for col in df.columns:
        if col not in categorical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 7. Drop rows where 'Player' is NaN
    if 'Player' in df.columns:
        df = df.dropna(subset=['Player'])

    # 8. Reset Index
    df = df.reset_index(drop=True)
    
    # Save Cleaned CSV
    df.to_csv(clean_path, index=False, encoding='utf-8-sig')
    log(f"Cleaned data saved to: {clean_path}")
    log(f"Final shape: {df.shape}")

    return clean_path


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    league_keys = list(LEAGUES.keys())
    for i, key in enumerate(league_keys):
        print(f"  [{i+1}] {LEAGUES[key]['name']}")

    choice = input("\nVotre choix (1-5) : ").strip()
    try:
        idx = int(choice) - 1
        assert 0 <= idx < len(league_keys)
        league_key = league_keys[idx]
    except (ValueError, AssertionError):
        print("Choix invalide.")
        sys.exit(1)

    step = input("Etape (1/2/3/all) : ").strip().lower()
    if step in ("1", "all"):
        scrape_stats(league_key)
    if step in ("2", "all"):
        scrape_photos(league_key)
    if step in ("3", "all"):
        path = clean_csv(league_key)
        if path:
            print(f"Fichier pret : {path}")
import pandas as pd
from DrissionPage import ChromiumPage, ChromiumOptions
import time
import os
import random

INPUT_CSV = "premier_league_players11.csv"
OUTPUT_CSV = "premier_league_players_with_photos.csv"

# ============================================================
# ANTI-BLOCK HELPERS
# ============================================================
def human_delay(min_s=2.5, max_s=5.5):
    time.sleep(random.uniform(min_s, max_s))

def long_pause(min_s=12.0, max_s=22.0):
    t = random.uniform(min_s, max_s)
    print(f"  Pause anti-blocage : {t:.1f}s...", flush=True)
    time.sleep(t)

def make_page(port=9333):
    co = ChromiumOptions()
    co.set_local_port(port)
    co.set_user_data_path(f"/tmp/drission_photos_{port}")
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
# PHOTO EXTRACTION
# ============================================================
def extract_photo_from_page(page):
    imgs = page.eles("t:img", timeout=3)
    for img in imgs:
        src = img.attr("src") or ""
        if "headshots" in src and "silhouette" not in src:
            return src
    return None


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60, flush=True)
    print("  FBref Player Photo Scraper (DrissionPage)", flush=True)
    print("="*60, flush=True)

    if not os.path.exists(INPUT_CSV):
        print(f"'{INPUT_CSV}' introuvable.", flush=True)
        return

    df = pd.read_csv(INPUT_CSV)

    url_cols = [c for c in df.columns if 'Player_URL' in str(c)]
    if not url_cols:
        print("Colonne 'Player_URL' introuvable.", flush=True)
        return
    url_col = url_cols[0]

    results = [""] * len(df)
    start_index = 0
    progress_path = "progress_" + OUTPUT_CSV
    if os.path.exists(progress_path):
        prog_df = pd.read_csv(progress_path)
        if 'Image_URL' in prog_df.columns:
            saved = list(prog_df['Image_URL'].fillna(""))
            start_index = len(saved)
            for i, v in enumerate(saved):
                results[i] = v
            print(f"Reprise a partir du joueur {start_index + 1}...", flush=True)

    name_cols = [c for c in df.columns if 'Player' in str(c) and 'URL' not in str(c)]

    import threading
    import queue

    NUM_THREADS = 5
    BASE_PORT = 9340

    q = queue.Queue()
    for i in range(start_index, len(df)):
        q.put(i)

    results_lock = threading.Lock()
    progress_lock = threading.Lock()
    completed = [0]

    print(f"Lancement de {NUM_THREADS} navigateurs DrissionPage en parallele...", flush=True)

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
                name = row[name_cols[0]] if name_cols else f"Joueur {i}"

                photo_url = ""
                if url and url not in ("nan", "Player_URL") and url.startswith("https://"):
                    print(f"[T{thread_id}][{i+1}/{len(df)}] {name}...", flush=True)
                    try:
                        safe_get(page, url)
                        human_delay(2.5, 5.5)
                        photo_url = extract_photo_from_page(page)
                        if not photo_url:
                            human_delay(2.0, 4.0)
                            photo_url = extract_photo_from_page(page)
                        print(f"  [T{thread_id}] -> {'Trouvee' if photo_url else 'Aucune'}", flush=True)
                    except Exception as e:
                        print(f"  [T{thread_id}] -> Erreur: {str(e)[:60]}", flush=True)

                with results_lock:
                    results[i] = photo_url if photo_url else ""

                with progress_lock:
                    completed[0] += 1
                    if completed[0] % 10 == 0:
                        temp_df = df.copy()
                        temp_df['Image_URL'] = results
                        temp_df.to_csv(progress_path, index=False)

                q.task_done()
        finally:
            page.quit()
            print(f"  [T{thread_id}] Navigateur ferme.", flush=True)

    threads = []
    for t_id in range(NUM_THREADS):
        t = threading.Thread(target=worker, args=(t_id,), daemon=True)
        threads.append(t)
        t.start()
        time.sleep(2)

    for t in threads:
        t.join()

    df['Image_URL'] = results

    if os.path.exists(progress_path):
        os.remove(progress_path)

    have_photos = df[df['Image_URL'].str.len() > 0]
    print(f"\nTermine ! Photos trouvees pour {len(have_photos)} joueurs sur {len(df)}.", flush=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"Sauvegarde sous : {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
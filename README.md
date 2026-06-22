# ⚽ Smart-Football-Analytics

> A full-stack interactive web platform for **multivariate statistical analysis** of football player data — powered by AI, live scraping, and modern visualizations.

Upload any CSV dataset or scrape live stats directly from **FBref**, then run **PCA**, **LDA**, or **Correspondence Analysis** in your browser — no coding required.

---

## ✨ Features

### 📐 Statistical Analysis
- **PCA (ACP)** — Normalized & non-normalized, Kaiser / Quality criterion, eigenvalues, correlation circle, individuals cloud, K-Means clustering
- **LDA (AFD)** — 3-class discriminant analysis, centroid classifier, confusion matrix, accuracy / precision / recall / F1
- **CA (AFC)** — Correspondence Analysis for contingency tables, row/column biplots *(coming soon)*

### 🤖 AI Agent (Groq LLaMA 3.3 70B)
- **Auto-Interpretation** — The agent reads your PCA/LDA results and generates a full statistical interpretation: axis meaning, partitions, atypical individuals, and conclusion
- **Conversational Chat** — Ask follow-up questions about your results in natural language. The agent has full context of your analysis (eigenvalues, variable coordinates, clusters)
- **Multi-model fallback** — Automatically switches between LLaMA 3.3 70B → LLaMA 3.1 70B → Mixtral 8x7B if a model is unavailable

### 🕷️ Live Data Scraping (FBref)
- Scrape **real-time player statistics** from FBref for 5 major European leagues:
  - 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League
  - 🇫🇷 Ligue 1
  - 🇪🇸 La Liga
  - 🇩🇪 Bundesliga
  - 🇮🇹 Serie A
- Downloads **player photos** automatically from official sources
- Full pipeline: **scrape stats → scrape photos → clean CSV** in one click
- Real-time progress logs streamed to the UI during scraping

### 🗂️ Interactive Data Editor
- Drag-and-drop CSV upload
- Filter rows with custom rules (`=`, `>`, `<`, `contains`)
- Select/deselect columns with chip UI
- Edit cell values directly in the browser before running analysis
- Add/remove rows and columns on the fly

### 📤 Tableau Export
- One-click export to **Tableau `.twbx`** workbook
- Embedded `.hyper` data extract with Individuals + Variables tables
- Pre-built dashboard with Scatter Plot + Correlation Circle sheets

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python · Flask |
| ML / Stats | NumPy · pandas · scikit-learn · scipy |
| Visualization | Plotly · Matplotlib |
| AI Agent | Groq API — LLaMA 3.3 70B |
| Web Scraping | requests · BeautifulSoup · FBref |
| BI Export | Tableau Hyper API |
| Frontend | HTML · CSS · Vanilla JS |

---

## 📊 Supported Analyses

| Method | Use Case | Input Data |
|--------|----------|------------|
| ACP | Dimensionality reduction, variable correlation | Continuous numeric |
| AFD | Group discrimination, classification | Numeric + categorical target |
| AFC | Categorical variable associations | Contingency table |

---

## 📁 Included Dataset

Ready-to-use **Premier League 2024/25 player dataset** — 535 players, 19 statistical features,
pre-processed for LDA with 3 position classes: **Defense · Milieu · Attaque**.
Scraped directly from FBref using the built-in scraper.

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/your-username/Smart-Football-Analytics.git
cd Smart-Football-Analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
echo GROQ_API_KEY=your_key_here > .env

# 4. Run the app
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📸 Screenshots

> *Add screenshots of the dashboard, PCA results, and AFD plots here*

---


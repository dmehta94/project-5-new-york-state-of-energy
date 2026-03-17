# New York State of Energy: Renewable Energy by 2030

*A group project from the General Assembly Data Science Bootcamp, January 2025*

**Team:** Muhammad Haseeb Anjum · Graham Haun · Melissa Marshall · Deval Mehta · Damar Shipp
**My role:** Technical lead and project manager — I coordinated the team's workflows, wrote the README and executive summary, independently researched and implemented the Time Series K-Means Clustering model, and troubleshot and integrated each team member's code contributions.

---

## What It Does

This project uses k-means clustering, time-series k-means clustering, and Prophet time-series forecasting to identify the optimal regions across New York State for solar and wind energy infrastructure — and then projects the State's energy consumption needs through 2030 to estimate how much capacity would be required.

We collected 20 years of daily weather data across 166 coordinate points, clustered those points into climatological zones, forecast weather trends through 2030, and overlaid projected county-level energy demand to produce a data-driven recommendation for the New York State Energy Planning Board.

---

## Why We Built This

New York State has committed to reducing greenhouse gas emissions to 85% of 1990 levels by 2050 under the [New York State Energy Plan](https://energyplan.ny.gov/). Meeting that target requires replacing carbon-based energy infrastructure with renewable alternatives — but the State spans dramatically different climate zones, from the Great Lakes effect regions of Western NY to the coastal weather patterns of Long Island. The question isn't just *whether* to build renewable infrastructure — it's *where*, and *what kind*.

This project was our final group project at GA bootcamp, completed just before our individual capstones. Our team chose it because it was a real, open policy problem with real publicly available data, and because it required us to push beyond what the bootcamp had taught us. Neither time-series clustering nor large-scale API data collection were part of the curriculum — I had to learn both independently mid-project.

**The practical output:** A map of New York State divided into wind-favorable and solar-favorable zones, with projected energy demand curves through 2030, designed to inform where the State should prioritize building new infrastructure.

---

## What I Learned

### Technical skills
- **Time Series K-Means Clustering with DTW:** The `tslearn` library's `TimeSeriesKMeans` with a soft-DTW metric was something I had to learn entirely from scratch. Standard k-means treats time as irrelevant; time-series k-means accounts for temporal patterns across the forecast horizon. Getting this to run on ~166 locations × 300+ months × 12+ features required careful data reshaping and understanding of how `tslearn` expects its input arrays.
- **Prophet at scale:** I applied Meta's Prophet model to two separate problems — forecasting 12 weather features per location (weather clustering) and forecasting energy load per county (~62 counties). Writing a clean abstraction that handled both use cases without duplicating logic taught me a lot about function design under varying input shapes.
- **API rate limiting in practice:** The Open-Meteo API has per-minute, per-hour, and per-day limits. I learned to detect rate limit errors from exception messages and apply the correct pause duration, rather than applying a blanket sleep. That distinction matters when you're making 1,000+ requests.
- **ETL pipeline design:** Managing three separate ETL pipelines (weather collection, load data extraction, county forecasting) with intermediate files between them forced me to think carefully about data contracts — what format each script expects to receive, and what format it promises to produce.

### Data science insights
- **Sparse data reveals itself late.** We discovered after collecting the weather data that the Open-Meteo API only returned complete multi-point data through 2010 — 2011 onward was available at only one coordinate. We hadn't caught this during our API testing phase. It's a reminder that validating data completeness should happen immediately after collection, not after you've built your entire pipeline on top of it.
- **A three-cluster solution didn't hold.** We started with the assumption that we'd find three clusters — solar, wind, and hydro. The silhouette score for three clusters (0.29) told us otherwise. Dropping to two clusters improved the score to 0.44, and feature engineering pushed it to 0.575. Sometimes the data doesn't confirm your hypothesis, and you have to follow the numbers.
- **Prophet's long-horizon limit is real.** The energy load projections showed consumption eventually reaching zero — physically impossible. Prophet models seasonal and trend components well over 3-5 year horizons, but it extrapolates trend linearly beyond that. Any time-series forecast extending 10+ years needs strong domain constraints to stay plausible.

### Software engineering practices
- **Shared constants belong in one place.** Midway through the project, two team members defined different versions of the zone-to-county mapping and the coordinate list. Consolidating these into module-level constants was a lesson I'll carry forward to every project.
- **Global mutable state is a trap.** The original `Energy_Load_ETL.py` used a `global load_df` that accumulated data across function calls. This worked until it didn't — order-dependent side effects are hard to debug and impossible to test cleanly. I restructured it to return DataFrames and concatenate explicitly.

### Unexpected learnings
- **Technical lead ≠ doing everything.** I came in with strong opinions about code quality and initially wanted to rewrite everything myself. I learned instead to write clear interface contracts, let team members implement their pieces, and do targeted code review rather than wholesale replacement. That preserved everyone's ownership and kept us on schedule.
- **Domain knowledge matters for feature selection.** The features that most improved our clustering score weren't the ones I would have guessed from the correlation matrix alone — the `wind_speed_index` and `temp_daylight_interaction` interaction features made the solar/wind split much cleaner. I had to read about what actually differentiates solar and wind viability before I understood which engineered features would be meaningful.

### Design decisions
- **Prophet over SARIMA for weather forecasting:** Prophet handles missing data, multiple seasonalities, and holiday effects without parameter tuning, which was critical when we had gaps in the weather data. SARIMA would have required careful order selection per feature per location — over a thousand models to tune. Prophet let us scale to that problem size.
- **Two clusters instead of three:** We explicitly considered forcing a third hydro cluster but found no well-separated hydro signal in the weather data. Hydroelectric viability depends more on river geography and water flow than on the weather variables we collected, so the clustering correctly found no third natural grouping.
- **Parallel K-Means on 12 threads:** The `TimeSeriesKMeans` model is computationally expensive with soft-DTW. We parallelized it across 12 threads (`n_jobs=12`), which requires a machine with at least 6 cores/12 threads. Users on less powerful hardware should reduce `n_jobs`.

---

## Quick Start

### Prerequisites
- Python 3.10+
- CPU with at least 6 cores / 12 threads (for Time Series K-Means; reduce `n_jobs` otherwise)

### Setup

```bash
git clone https://github.com/dmehta94/project-5-new-york-state-of-energy.git
cd project-5-new-york-state-of-energy

python -m venv venv
source venv/Scripts/activate  # Windows / GitBash
# source venv/bin/activate    # macOS / Linux

pip install -r requirements.txt
```

### Run the pipeline

The pipeline has three stages that must run in order:

```bash
# 1. Collect 20 years of weather data (takes several hours — API rate limited)
python code/Weather_Data_Collection.py

# 2. Extract and compile NYISO load data (requires pre-downloaded zip archives)
python code/Energy_Load_ETL.py

# 3. Run county-level load forecasts (requires Analysis.ipynb pivoting step first)
python code/County_Forecasting_ETL.py

# 4. Run the full analysis
jupyter notebook code/Analysis.ipynb
```

**Note:** The weather data CSV (~500MB) and load data zip archives are not included in the repository due to size. See Data Sources below for download links.

---

## Data Sources

| Dataset | Source | Description |
|---|---|---|
| Daily weather data | [Open-Meteo Archive API](https://open-meteo.com/) | 12 daily variables, 166 coordinate points, 2005–2024 |
| Energy load by zone | [NYISO](https://www.nyiso.com/load-data) | Hourly zone-level load data, 2008–2024 |
| County population | [NY State Open Data](https://data.ny.gov/Government-Finance/Annual-Population-Estimates-for-New-York-State-and/krt9-ym2k/about_data) | Annual county population estimates |

---

## Technical Details

### Stack

| Library | Purpose |
|---|---|
| `prophet` | Time-series forecasting for weather features and county load |
| `tslearn` | Time Series K-Means Clustering with soft-DTW metric |
| `sklearn` | Standard K-Means, PCA, StandardScaler, silhouette scoring |
| `pandas`, `numpy` | Data manipulation and feature engineering |
| `matplotlib`, `seaborn` | Visualization |
| `openmeteo_requests`, `requests_cache` | Open-Meteo API client with caching |
| `zipfile`, `os` | NYISO archive extraction |

### Key functions

**`Weather_Data_Collection.py`**
- `fetch_weather_data(lat, lon)` — Retrieves daily weather records for one coordinate from the Open-Meteo Archive API
- `collect_weather_data(coordinates)` — Iterates over all 166 coordinates with rate limiting, retry logic, and partial-failure tracking

**`Energy_Load_ETL.py`**
- `unzip_all_archives(base_path, start_year, end_year)` — Extracts all NYISO zip archives from year-named subdirectories
- `compile_data(file_name, base_path)` — Reads one NYISO CSV and aggregates to daily average load per zone
- `compile_all_data(base_path)` — Calls `compile_data` across all extracted CSVs and concatenates results

**`County_Forecasting_ETL.py`**
- `forecast_county(data, county)` — Trains Prophet on one county's load history and forecasts through 2050
- `run_county_forecasts(data)` — Runs the above for all ~62 counties and merges into a wide DataFrame
- `extract_date_range(file_name, start_date, end_date)` — Filters the forecast CSV to the analysis window

**`Analysis.ipynb`**
- `preprocess_and_forecast_prophet(data, forecast_end_year)` — Aggregates weather data to monthly frequency, runs Prophet on each feature per location, returns a 3D array `(n_locations, n_months, n_features)` for time-series clustering
- `cluster_summary(forecasted_data, clusters, feature_names)` — Computes average feature values per cluster for interpretability
- `plot_clusters_on_map(locations_df, cluster_assignments)` — Plots cluster assignments as a geographic scatter over NY State coordinates

### Coordinate selection
We selected 166 coordinate points to cover all 62 counties, typically three points per county: the county centroid plus two secondary points. This spatial resolution gave us enough variation to identify meaningful climatological subregions without exceeding the Open-Meteo free-tier daily request limit.

---

## Findings

New York State shows a clear two-cluster structure: a wind-favorable cluster covering essentially all of upstate NY, and a solar-favorable cluster concentrated in the southeastern corner — the NYC metro area, Long Island, and the lower Hudson Valley. The split is roughly upstate vs. downstate, with the solar cluster identified by lower latitudes (below ~42°N) and longitudes east of roughly -74.5°.

The exploratory k-means model on 2020 weather data reached a silhouette score of 0.575 after feature engineering and PCA tuning. The time-series k-means model — which accounts for temporal patterns across the full forecast horizon — returned a silhouette score of 0.434, lower than the 2020 snapshot. The presentation notes this as evidence of climate-driven cluster drift over time: the boundary between wind and solar zones is shifting, with the solar zone expanding northward.

Statewide energy consumption declined from roughly 19,000 MW/year in 2005 to a projected ~16,000 MW/year by 2030 — a 16% decrease — with strong seasonal peaks in winter and summer. Kings County is the top energy consumer in the state, averaging approximately 1,750 megawatts per day, followed by Queens, New York, Suffolk, and Nassau counties — all downstate. The projected forecast maintains the seasonal pattern but shows continued gradual decline.

Our cost analysis (see `cost_analysis.md`) projects that solar and wind infrastructure would pay back within 6–12 years depending on technology, with federal incentives (30% ITC for solar, $25/MWh PTC for wind) accelerating returns. Long-term, the transition is projected to create 250,000+ jobs and save the State billions.

---

## Limitations

- **Sparse post-2010 weather data:** The Open-Meteo API returned complete multi-point data only through 2010. The 2011–2024 records cover only a single coordinate. Our clustering results for the forecast period should be interpreted with this data quality issue in mind.
- **Prophet's long-horizon extrapolation:** The energy consumption projections show demand approaching zero by 2030, which is not physically realistic. Prophet captures recent trend and seasonality well but extrapolates trend linearly over long horizons without domain constraints.
- **No hydro cluster:** We could not identify a well-defined hydroelectric cluster from weather variables alone. Hydro viability depends primarily on river geography and water flow, not the weather features we collected.
- **Zone-to-county population weighting:** We approximated county load by distributing zonal NYISO load data by population share. This is a reasonable proxy but doesn't account for industrial vs. residential load differences across counties.
- **Static coordinate grid:** The 166 coordinate points were manually chosen. A more rigorous spatial sampling approach (e.g., stratified random sampling by county area) would improve representativeness.

---

## Credits

**Team contributions:**
- **Deval Mehta** — Technical lead, project manager, Time Series K-Means model, README and executive summary, code coordination and troubleshooting
- **Muhammad Haseeb Anjum** — Energy load ETL pipeline, county load forecasting ETL, load data analysis and visualization
- **Graham Haun** — Weather data collection script, initial data cleaning, exploratory k-means clustering
- **Melissa Marshall** — EDA, feature engineering, data visualization, EDA insights slide
- **Damar Shipp** — Cost analysis (`cost_analysis.md`), findings and implications

**Instructors:** Matt Brems and Asha Mathis (General Assembly) provided guidance during the project.

**AI assistance (original project):** Several team members used ChatGPT during development. I used it as a learning tool for the Time Series K-Means work: it helped me set up the initial pipeline, explained the available options in `TimeSeriesKMeans`, walked me through evaluation metric choices, clarified the difference between DTW and soft-DTW, introduced me to `pmdarima` and `tqdm`, and helped me think through automating ARIMA order selection. Muhammad used it similarly to learn Prophet for the load forecasting work. Graham used it to troubleshoot the weather data collection script. In all cases, ChatGPT was a learning and debugging aid; all analytical decisions, modeling choices, and findings are the team's own. Claude (Anthropic) assisted with post-bootcamp code cleanup, standardization, docstrings, and README writing.

---

## License

MIT License — see `LICENSE` for details.

**Contact:** Deval Mehta · [GitHub @dmehta94](https://github.com/dmehta94)
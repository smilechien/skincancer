# skincancer
# Rasch-CAT for Skin Cancer

A Flask-based web application for a 30-item skin-cancer / melanoma-risk questionnaire using a Rasch Partial Credit Model (PCM) computerized adaptive testing (CAT) workflow.

The app supports item-by-item CAT administration, automatic CAT sequential simulation from sample data, non-CAT administration, voice practice, CAT vs non-CAT simulation, Wright Map inspection, KIDMAP interpretation, category probability curves, and downloadable PNG charts.

> **Important medical note:** This app provides a Rasch-estimated skin-cancer / melanoma-risk classification for questionnaire research and educational use. It is **not** a clinical diagnosis, cancer screening test, biopsy result, or melanoma staging tool.

---

## Main features

- **Rasch PCM CAT engine**
  - Item-specific step thresholds
  - EAP posterior theta estimation
  - Posterior standard error (SE)
  - PCM item information based on expected score variance
  - Adaptive item selection by maximum item information

- **30-item skin-cancer risk item bank**
  - Constitutional risk items, such as skin colour, eye colour, hair colour, freckles, and moles
  - UV-exposure history, such as sunbed use, sunburn history, and outdoor sun hours
  - Family and medical history items
  - Sun-protection behaviour items
  - Melanoma-status / class outcome is excluded from CAT scoring

- **Cronbach alpha-based stopping criterion**
  - The homepage computes the default CAT stopping SE from simulated or sample response data:

    ```text
    Stop SE = theta SD × sqrt(1 − Cronbach alpha)
    ```

  - The computed SE is shown on the homepage and can still be edited by the user.

- **CAT sequential simulation mode**
  - Uses the homepage **Starting theta**
  - Selects candidate persons from the sample data whose estimated theta is closest to the selected starting theta
  - Randomly selects one person from the candidate pool
  - Runs the CAT automatically using that person's stored responses
  - Skips manual item-by-item answering and goes directly to the completed CAT results page

- **Visual dashboards**
  - Homepage Wright Map
  - Homepage reference-person KIDMAP
  - Result trend chart
  - KIDMAP-style dashboard
  - Category probability curves (CPC)

- **PNG export**
  - Download homepage Wright Map as PNG
  - Download homepage KIDMAP as PNG
  - Download CAT result trend chart as PNG
  - Download result KIDMAP as PNG
  - Download CPC as PNG

- **Question images**
  - Each item can be linked to a PNG image stored in the `pic/` folder
  - The `response_category.csv` file uses the `link` column to connect each item to its picture

---

## Repository structure

```text
.
├── app.yaml
├── raschcatpcm.py
├── README.md
├── replay_bundle.zip
├── requirements.txt
├── response_category.csv
└── pic/
    ├── Q01_Sex_CR.png
    ├── Q02_How_would_you_rate_your_natural_skin_colour_on_areas_never_e.png
    ├── ...
    └── Q30_Thinking_about_ALL_of_the_times_when_you_were_outside_in_the.png
```

---

## Required files

### `raschcatpcm.py`

Main Flask application. It contains:

- Rasch PCM item-bank loader
- CAT administration logic
- EAP posterior estimation
- CAT sequential simulation
- Cronbach alpha-based SE stopping rule
- KIDMAP risk classification
- SVG chart generation
- PNG download support
- Google App Engine-compatible Flask entry point

The Flask object must be named:

```python
app = Flask(__name__)
```

### `replay_bundle.zip`

Main data bundle loaded by the app. It should include the required Rasch and response files, such as:

```text
response_category.csv
fixed_item_delta.csv
person_estimates.csv
item_estimates.csv
zscore.csv
item_step_delta.csv
metadata.json
pic/
```

The app can also read a root-level `response_category.csv` if present.

### `response_category.csv`

Defines item wording, item links, response categories, and step thresholds.

Expected key columns:

```text
no
link
delta
item
item2
option
option2
Title
Step
```

Typical example:

```text
no: 1
link: Q01_Sex_CR.png
item: 性別
item2: Gender
option: 0=女性, 1=男性
option2: 0=Female, 1=Male
Title: Rasch-CAT for Skin Cancer
Step: 0.000
```

### `pic/`

Stores the item-level PNG images. The image filename in the `link` column is resolved automatically as:

```text
pic/<filename>.png
```

For example:

```text
response_category.csv link = Q01_Sex_CR.png
resolved file path       = pic/Q01_Sex_CR.png
```

---

## Installation

Create and activate a Python virtual environment:

```bash
python -m venv venv
```

Windows:

```bat
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Suggested `requirements.txt`:

```text
Flask
gunicorn
numpy
pandas
gTTS
```

`gTTS` is optional. If unavailable, the app still runs, but MP3 voice features may be disabled.

---

## Run locally

```bash
python raschcatpcm.py
```

Then open:

```text
http://127.0.0.1:8080
```

If the app uses a different port, check the terminal message after startup.

---

## App modes

### 1. CAT

Standard item-by-item adaptive testing mode.

Workflow:

```text
Start CAT → answer current item → update theta and SE → select next item → stop by SE or maximum items
```

### 2. CAT sequential simulation

Automatic completed CAT mode based on sample data.

Workflow:

```text
Set Starting theta
Set simulated/candidate persons N
Select candidate persons closest to Starting theta
Randomly choose one candidate
Run CAT automatically using stored responses
Show completed CAT result page
```

This mode is useful for demonstrations because it avoids manual answering and immediately shows the completed CAT outputs.

### 3. non-CAT

Administers items in fixed item-number order.

### 4. Voice practice

Shows one item at a time with linked picture and MP3 audio.

### 5. CAT vs non-CAT(1)

Generates one full-answer pattern and compares full non-CAT estimation with repeated CAT administrations.

### 6. CAT vs non-CAT(n)

Simulates multiple independent persons and compares CAT and full non-CAT estimates.

---

## CAT stopping rule

The homepage displays an alpha-based default stopping criterion:

```text
Stop CAT when posterior SE ≤ Cronbach’s alpha-based default
```

The formula is:

```text
Stop SE = theta SD × sqrt(1 − Cronbach alpha)
```

For example:

```text
theta SD = 1.00
Cronbach alpha = 0.9879

Stop SE = 1.00 × sqrt(1 − 0.9879)
        ≈ 0.110
```

A smaller SE requires more precise measurement and usually requires more CAT items. If the requested SE is too strict for the 30-item bank, CAT will stop at the maximum item limit.

---

## KIDMAP risk classification

The Results page classifies skin-cancer / melanoma risk using the **overall KIDMAP person measure**, not individual item bubbles.

The default rule assumes that theta is centered around 0 and has approximately SD = 1:

```text
theta < -0.5       = low / mild risk
-0.5 to 0.5        = average / moderate risk
0.5 to 1.5         = high risk
theta > 1.5        = very high risk
```

Example:

```text
KIDMAP — Measure -0.07 (SE 0.24)
Classification: average / moderate risk
```

The KIDMAP item bubbles are used to inspect unexpected item responses:

```text
ZSTD > +2 = unexpectedly high-risk response
ZSTD < -2 = unexpectedly low-risk or protective response
```

Again, this is a risk classification based on questionnaire measurement, not a clinical diagnosis.

---

## PNG downloads

The app supports browser-side PNG export of SVG charts.

Homepage downloads:

```text
wright_map_homepage.png
reference_person_kidmap_homepage.png
```

Results page downloads:

```text
cat_result_trend_chart.png
kidmap_style_dashboard.png
category_probability_curves.png
```

If a PNG download button does not respond, restart the Flask app and refresh the browser page to ensure the latest JavaScript is loaded.

---

## Google App Engine deployment

Example `app.yaml`:

```yaml
service: raschcatpcmskin
runtime: python311
env: standard
entrypoint: gunicorn -b :$PORT raschcatpcm:app --timeout 120

automatic_scaling:
  max_instances: 1
```

Deploy:

```bash
gcloud app deploy app.yaml
```

Check deployed services:

```bash
gcloud app services list
```

Check deployed versions:

```bash
gcloud app versions list
```

---

## Recommended `.gcloudignore`

Use this to deploy only required files:

```text
# Ignore everything first
*

# Keep required root files
!app.yaml
!raschcatpcm.py
!README.md
!README.txt
!replay_bundle.zip
!requirements.txt
!response_category.csv

# Keep picture folder and all images
!pic/
!pic/**

# Ignore common unnecessary files
.git/
.gitignore
__pycache__/
*.pyc
*.pyo
*.pyd
venv/
env/
.venv/
.DS_Store
Thumbs.db
```

Preview files to be uploaded:

```bash
gcloud meta list-files-for-upload
```

---

## Data preparation notes

Before using the app as a formal Rasch CAT instrument:

1. Recode nominal categories into risk-ordered groups where appropriate.
2. Collapse sparse adjacent categories when PCM step thresholds are disordered.
3. Refit the PCM after category revision.
4. Export the updated item difficulties and item-specific step thresholds.
5. Ensure the `response_category.csv` category wording matches the final scoring.
6. Exclude outcome variables, such as melanoma status, from the CAT item bank.

---

## Limitations

- The app estimates relative questionnaire-based skin-cancer / melanoma risk.
- It does not replace clinical skin examination, biopsy, diagnosis, or staging.
- Rasch PCM validity depends on item fit, ordered thresholds, category functioning, and dimensionality.
- Cronbach alpha is used only to derive a practical SE stopping criterion; it does not prove unidimensional Rasch model fit.
- PNG export is browser-side and may vary slightly across browsers.

---

## Suggested citation text

```text
Rasch-CAT for Skin Cancer is a Flask-based web application implementing a 30-item Rasch Partial Credit Model computerized adaptive testing workflow for skin-cancer / melanoma-risk questionnaire assessment.
```

---

## License

Add your preferred license here, for example:

```text
MIT License
```

---

## Author

Developed for Rasch PCM-based computerized adaptive testing of skin-cancer / melanoma-risk questionnaire data.

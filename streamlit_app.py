"""Streamlit thesis demo for AAC adoption analysis."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import altair as alt

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.dashboard.data import (  # noqa: E402
    best_model_rows,
    build_profile_prediction_record,
    build_prediction_record,
    load_diagnostic,
    load_optional_csv,
    load_summary,
    load_table,
    local_shap_explanations,
    predict_from_record,
    profile_global_shap_reasons,
    similar_historical_cases,
    visibility_need_from_prediction,
)
from aac_adoption.dashboard.story import (  # noqa: E402
    approach_comparison_rows,
    decision_sankey,
    story_cards,
    workflow_dot,
)
from aac_adoption.dashboard.trust_page import render_trust_and_limits  # noqa: E402

# ---------------------------------------------------------------------------
# Methodological language constants - used consistently across all SHAP views
# ---------------------------------------------------------------------------
SHAP_DISCLAIMER = (
    "SHAP values explain how features contributed to this model's prediction. "
    "They do not prove that changing a feature would causally change adoption probability."
)
CAUSAL_WARNING = (
    "**Methodological note:** Changing fields in this form shows how the trained model "
    "reacts to different input records. It does not prove that changing a real animal's "
    "characteristic would change its adoption outcome."
)
INTAKE_ONLY_NOTE = (
    "This model uses only information available **at intake time**. "
    "It cannot account for changes after an animal enters the shelter."
)


TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SUMMARY_DIR = PROJECT_ROOT / "reports" / "summary"
DIAGNOSTICS_DIR = PROJECT_ROOT / "reports" / "diagnostics"
MODELS_DIR = PROJECT_ROOT / "models" / "advanced"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"

THESIS_REPORT_OPTIONS = {
    "docs/target_definitions.md": "Target Definitions & Outcome Mappings",
    "reports/summary/external_validity_limitations.md": "External Causal Validity & Generalisation Limits",
    "reports/summary/breed_color_justification.md": "Breed and Coat Colour Feature Engineering Justification",
    "reports/summary/descriptive_baseline_comparison.md": "Machine Learning vs. Descriptive Non-ML Baselines",
    "reports/summary/h1_interpretation.md": "H1 - Intake Profile & Causal Context",
    "reports/summary/h2_interpretation.md": "H2 - Seasonality & Intake Dynamics",
    "reports/summary/h3_interpretation.md": "H3 - Age and Time-to-Outcome Timing",
    "reports/summary/h4_interpretation.md": "H4 - Coat Colour (Black/Dark Animal Syndrome Check)",
    "reports/summary/h5_interpretation.md": "H5 - COVID-Period Population Shift and Volume Impact",
    "reports/summary/hypothesis_evidence_matrix.md": "Hypothesis Evidence Matrix Summary",
    "reports/summary/leakage_audit.md": "Data Leakage Audit & Control Log",
    "reports/summary/matching_logic_examples.md": "Propensity Score Matching Validation Examples",
    "reports/summary/model_evidence_pack.md": "Narrative Model Evidence & Key Findings Pack",
    "reports/summary/subgroup_reliability.md": "Subgroup Reliability & Underrepresented Cohorts",
    "reports/summary/final_model_selection.md": "Final Model Architecture Selection",
    "reports/summary/threshold_selection.md": "Optimal Classification Threshold & Utility Analysis",
    "reports/summary/calibration_interpretation.md": "Model Probability Calibration Interpretation",
    "reports/summary/model_reliability_red_flags.md": "Operational Risk & Model Reliability Red Flags",
    "reports/summary/data_audit.md": "Data Pipeline Audit & Attrition Logging",
    "reports/summary/environment_snapshot.md": "Reproducibility Snapshot & Environment Info",
}

LANGUAGES = {
    "English": "en",
    "Polski": "pl",
}

PL = {   'AAC Adoption Thesis Demo': 'Wizualizacja wyników pracy magisterskiej: '
                                'analiza predykcyjna adopcji w AAC',
    'Artifact-driven dashboard for model results, hypothesis signals, and model sensitivity checks.': 'Interaktywny '
                                                                                                      'system '
                                                                                                      'wizualizacji '
                                                                                                      'modeli '
                                                                                                      'klasyfikacyjnych, '
                                                                                                      'weryfikacji '
                                                                                                      'hipotez '
                                                                                                      'oraz '
                                                                                                      'analizy '
                                                                                                      'wrażliwości.',
    'Language': 'Język',
    'Executive Overview': 'Przegląd analityczny',
    'Story Mode': 'Analiza ścieżek (Story Mode)',
    'Animal Stories': 'Analiza przypadków (Case Studies)',
    'Model Quality': 'Ewaluacja predykcyjna modelu',
    'Trust & Limits': 'Analiza wiarygodności i ograniczeń',
    'Interpretability': 'Interpretowalność (XAI)',
    'Risk Explorer': 'Moduł ewaluacji ryzyka',
    'Hypothesis Lab': 'Weryfikacja hipotez badawczych',
    'Campaign Finder': 'Identyfikacja celów interwencji promocyjnych',
    'Adoption Timeline': 'Analiza przeżycia (Oś czasu adopcji)',
    'Artifacts': 'Zestawienie artefaktów badawczych',
    'Context Data': 'Zmienne środowiskowe i kontekstowe',
    'Run the training and analysis pipeline to populate model comparison outputs.': 'Uruchom '
                                                                                    'potok '
                                                                                    'trenowania '
                                                                                    'i '
                                                                                    'analizy '
                                                                                    '(pipeline), '
                                                                                    'aby '
                                                                                    'wygenerować '
                                                                                    'wyniki '
                                                                                    'porównawcze '
                                                                                    'modeli.',
    'Missing figure': 'Brakujący artefakt graficzny',
    'Data-to-Decision Story': 'Transformacja danych w sygnały decyzyjne',
    'How raw shelter records become thesis evidence and practical shelter-facing signals.': 'Proces '
                                                                                            'transformacji '
                                                                                            'surowych '
                                                                                            'danych '
                                                                                            'schroniskowych '
                                                                                            'w '
                                                                                            'empiryczny '
                                                                                            'materiał '
                                                                                            'badawczy '
                                                                                            'oraz '
                                                                                            'analityczne '
                                                                                            'sygnały '
                                                                                            'decyzyjne.',
    'Approach Comparison': 'Analiza porównawcza podejść analitycznych',
    'Analytical layer': 'Warstwa analityczna',
    'Story weight': 'Waga przypadku',
    'Probability Trust': 'Kalibracja i wiarygodność predykcji',
    'When model says 70% adoption chance, does reality agree?': 'Weryfikacja '
                                                                'stopnia '
                                                                'zgodności '
                                                                'prognozowanego '
                                                                'prawdopodobieństwa '
                                                                'adopcji z '
                                                                'rzeczywistą '
                                                                'częstością '
                                                                'zdarzeń.',
    'Calibration curve': 'Krzywa kalibracyjna (Reliability Diagram)',
    'Long-stay Risk': 'Estymacja ryzyka wydłużonego pobytu',
    'Which animals look adoptable but may wait longer?': 'Identyfikacja kohort '
                                                         'o wysokim '
                                                         'prawdopodobieństwie '
                                                         'adopcji przy '
                                                         'jednoczesnym '
                                                         'podwyższonym ryzyku '
                                                         'wydłużonego czasu '
                                                         'oczekiwania.',
    'Model Failure Modes': 'Charakterystyka błędów modelu (Failure Modes)',
    'Where do false negatives and large LOS errors cluster?': 'Identyfikacja '
                                                              'klastrów '
                                                              'fałszywie '
                                                              'negatywnych '
                                                              'predykcji oraz '
                                                              'wariancji w '
                                                              'estymacji czasu '
                                                              'pobytu.',
    'Which cohorts may deserve targeted visibility?': 'Identyfikacja '
                                                      'subpopulacji '
                                                      'wymagających '
                                                      'ukierunkowanych działań '
                                                      'promocyjnych.',
    'Similar Cases': 'Analiza historycznych przypadków referencyjnych',
    'What happened historically to animals like this one?': 'Analiza rozkładu '
                                                            'wyników dla '
                                                            'historycznych '
                                                            'wektorów o '
                                                            'wysokim stopniu '
                                                            'prawdopodobieństwa/podobieństwa.',
    'Real-life Shelter Questions': 'Zastosowanie operacyjne w warunkach '
                                   'schroniskowych',
    'Animal Journey Cards': 'Karty analizy ścieżki pobytu',
    'Run `python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv` to populate animal stories.': 'Uruchom '
                                                                                                                               '`python '
                                                                                                                               'scripts/generate_animal_research.py '
                                                                                                                               '--data '
                                                                                                                               'data/processed/modeling_dataset.csv`, '
                                                                                                                               'aby '
                                                                                                                               'wygenerować '
                                                                                                                               'analizę '
                                                                                                                               'przypadków '
                                                                                                                               'zwierząt.',
    'Animal profile': 'Profil analizowanego obiektu',
    'Similar records': 'Zbieżne wektory (N)',
    'Adoption rate': 'Odsetek pozytywnych adopcji (%)',
    'Median days to outcome': 'Mediana czasu do rozstrzygnięcia (dni)',
    'Visibility need': 'Ewaluacja potrzeby ekspozycji',
    'Profile': 'Charakterystyka obiektu',
    'has recorded name': 'zarejestrowane imię',
    'no recorded name': 'brak zarejestrowanego imienia',
    'Transfer rate': 'Odsetek transferów zewnętrznych (%)',
    'Return-to-owner rate': 'Odsetek powrotów do pierwotnego właściciela (%)',
    'Euthanasia rate': 'Wskaźnik eutanazji (%)',
    'Model View for This Journey': 'Wynik predykcji dla zadanego wektora',
    'Run `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv` to add representative CatBoost predictions to journey cards.': 'Uruchom '
                                                                                                                                                      '`python '
                                                                                                                                                      'scripts/train_advanced.py '
                                                                                                                                                      '--data '
                                                                                                                                                      'data/processed/modeling_dataset.csv`, '
                                                                                                                                                      'aby '
                                                                                                                                                      'włączyć '
                                                                                                                                                      'ewaluację '
                                                                                                                                                      'modelem '
                                                                                                                                                      'CatBoost '
                                                                                                                                                      'do '
                                                                                                                                                      'widoku '
                                                                                                                                                      'profili.',
    'Predicted adoption chance': 'Estymowane prawdopodobieństwo adopcji',
    'Predicted wait': 'Estymowany czas oczekiwania',
    'Predicted Time to Any Outcome': 'Estymowany czas do rozstrzygnięcia',
    'Model visibility label': 'Klasyfikacja zapotrzebowania na ekspozycję',
    'Length-of-stay bucket': 'Kategoryzacja estymowanej długości pobytu',
    'Representative model record': 'Referencyjny wektor cech w modelu',
    'high visibility need': 'wysoki priorytet ekspozycji',
    'medium visibility need': 'umiarkowany priorytet ekspozycji',
    'standard visibility': 'standardowy priorytet ekspozycji',
    'low': 'niska',
    'medium': 'umiarkowana',
    'high': 'wysoka',
    'Similar Historical Cases': 'Historyczne profile referencyjne',
    'No similar historical cases found for this representative card.': 'Brak '
                                                                       'historycznych '
                                                                       'obserwacji '
                                                                       'o '
                                                                       'wystarczającym '
                                                                       'stopniu '
                                                                       'zbieżności '
                                                                       'cech '
                                                                       'dla '
                                                                       'analizowanego '
                                                                       'wektora.',
    'Top SHAP Reasons': 'Analiza istotności cech lokalnych (SHAP)',
    'Run `python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap` to populate SHAP reasons.': 'Uruchom '
                                                                                                                                        '`python '
                                                                                                                                        'scripts/generate_diagnostics.py '
                                                                                                                                        '--data '
                                                                                                                                        'data/processed/modeling_dataset.csv '
                                                                                                                                        '--include-shap`, '
                                                                                                                                        'aby '
                                                                                                                                        'wygenerować '
                                                                                                                                        'wektory '
                                                                                                                                        'wartości '
                                                                                                                                        'SHAP.',
    'Model-wide SHAP signals mapped onto this animal profile; associations, not causes.': 'Globalne '
                                                                                          'wartości '
                                                                                          'dyskryminacyjne '
                                                                                          '(SHAP) '
                                                                                          'aproksymowane '
                                                                                          'dla '
                                                                                          'analizowanego '
                                                                                          'wektora; '
                                                                                          'wskazują '
                                                                                          'korelacje, '
                                                                                          'bez '
                                                                                          'implikacji '
                                                                                          'kauzalnych.',
    'Local CatBoost SHAP values for the representative journey record; associations, not causes.': 'Lokalne '
                                                                                                   'miary '
                                                                                                   'SHAP '
                                                                                                   '(CatBoost) '
                                                                                                   'dla '
                                                                                                   'analizowanego '
                                                                                                   'wektora; '
                                                                                                   'determinują '
                                                                                                   'asocjacje '
                                                                                                   'bez '
                                                                                                   'domniemania '
                                                                                                   'związków '
                                                                                                   'przyczynowo-skutkowych.',
    'Key Animal Contrasts': 'Kluczowe dysproporcje statystyczne',
    'Contrast': 'Rozbieżność (Contrast)',
    'Animal group': 'Kategoria podmiotu',
    'Largest animal archetypes': 'Dominujące klastry fenotypowe',
    'Animal profiles needing visibility or support': 'Profile wymagające '
                                                     'podwyższonej ekspozycji '
                                                     'lub specjalistycznego '
                                                     'wsparcia',
    'Vulnerable Profiles': 'Kohorty o podwyższonym poziomie ryzyka '
                           '(Vulnerable)',
    'Health and Behavior Support Profiles': 'Profile specyficznych interwencji '
                                            'medycznych i behawioralnych',
    'Classification Table': 'Macierz wyników klasyfikacyjnych',
    'Regression Table': 'Macierz błędów regresyjnych',
    'Probability Trust Meter': 'Analiza wiarygodności (Calibration / Trust)',
    'Run `python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv` to populate calibration diagnostics.': 'Uruchom '
                                                                                                                                    '`python '
                                                                                                                                    'scripts/generate_diagnostics.py '
                                                                                                                                    '--data '
                                                                                                                                    'data/processed/modeling_dataset.csv`, '
                                                                                                                                    'aby '
                                                                                                                                    'wyliczyć '
                                                                                                                                    'metryki '
                                                                                                                                    'kalibracyjne.',
    'Mean predicted probability': 'Średnie estymowane prawdopodobieństwo',
    'Observed adoption rate': 'Empiryczny odsetek adopcji',
    'Reliability Figures': 'Wizualizacja charakterystyk niezawodności',
    'Advanced model ROC curve': 'Krzywa ROC (Receiver Operating '
                                'Characteristic) modelu docelowego',
    'Advanced model precision-recall curve': 'Krzywa PR (Precision-Recall) '
                                             'modelu docelowego',
    'Probability calibration': 'Krzywa kalibracji prawdopodobieństw '
                               '(Reliability Curve)',
    'Regression predicted vs actual': 'Analiza wariancji: przewidywania '
                                      'regresora a wartości empiryczne',
    'Model Evidence Pack': 'Rozszerzone dossier ewaluacji modelu',
    'Run `python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv` to populate trust and limits artifacts.': 'Uruchom '
                                                                                                                                         '`python '
                                                                                                                                         'scripts/generate_evidence_pack.py '
                                                                                                                                         '--data '
                                                                                                                                         'data/processed/modeling_dataset.csv`, '
                                                                                                                                         'aby '
                                                                                                                                         'wygenerować '
                                                                                                                                         'estymatory '
                                                                                                                                         'zaufania '
                                                                                                                                         'analitycznego.',
    'Metric Confidence Intervals': 'Przedziały ufności dla metryk '
                                   'ewaluacyjnych',
    'Metric': 'Metryka',
    'Bootstrap interval': 'Estymator przedziałowy (Bootstrap CI)',
    'Cohort Reliability Limits': 'Analiza pogorszenia parametrów predykcyjnych '
                                 'w subpopulacjach',
    'Calibration gap': 'Błąd kalibracji (Calibration Gap)',
    'Cohort value': 'Kwantyfikacja dla subpopulacji',
    'Subgroup Explorer': 'Dekonstrukcja metryk w subpopulacjach',
    'Reliability subgroup': 'Rozpatrywana podgrupa demograficzna',
    'Where the Model Struggles': 'Obszary ograniczonej zdolności predykcyjnej',
    'Subgroup Metric Intervals': 'Analiza przedziałowa metryk podgrup',
    'Time-to-Adoption Milestones': 'Punkty krytyczne czasu przebywania w '
                                   'schronisku',
    'Milestone subgroup': 'Podgrupa ewaluowana (Survival Analysis)',
    'Adopted animals (%)': 'Skumulowany odsetek adopcji (%)',
    'Milestone': 'Kwantyl czasowy',
    'Value': 'Miara empiryczna',
    'Records': 'Liczba obserwacji (N)',
    'Adoptions': 'Zrealizowane adopcje',
    'Adopted by day': 'Rozstrzygnięte do (dzień)',
    'Animal Journey Evidence Examples': 'Empiryczna weryfikacja predykcji na '
                                        'tle przypadków historycznych',
    'SHAP Global Explanations': 'Globalna analiza struktury ważności cech '
                                '(SHAP)',
    'SHAP values describe factors associated with model predictions, not causal effects.': 'Miary '
                                                                                           'SHAP '
                                                                                           'kwantyfikują '
                                                                                           'relacyjną '
                                                                                           'istotność '
                                                                                           'predyktorów; '
                                                                                           'nie '
                                                                                           'uprawniają '
                                                                                           'one '
                                                                                           'do '
                                                                                           'konkluzji '
                                                                                           'o '
                                                                                           'mechanizmach '
                                                                                           'przyczynowo-skutkowych.',
    'Classification SHAP summary': 'Struktura ważności predyktorów: model '
                                   'klasyfikacyjny',
    'Regression SHAP summary': 'Struktura ważności predyktorów: model '
                               'regresyjny',
    'Feature Family Scores': 'Znaczenie predykcyjne zagregowanych klas cech',
    'Sum mean absolute SHAP': 'Zsumowana bezwzględna wartość aproksymacji SHAP',
    'Feature family': 'Kategoria predyktorów',
    'Run diagnostics with `--include-shap` to populate interpretation artifacts.': 'Zainicjuj '
                                                                                   'system '
                                                                                   'z '
                                                                                   'parametrem '
                                                                                   '`--include-shap` '
                                                                                   'dla '
                                                                                   'pełnej '
                                                                                   'dekompozycji '
                                                                                   'XAI.',
    'Risk Threshold Simulator': 'Analiza wrażliwości na przesunięcie progu '
                                'klasyfikacyjnego (Decision Threshold '
                                'Simulator)',
    'Run diagnostics to populate threshold tradeoffs.': 'Uruchom rutyny '
                                                        'diagnostyczne w celu '
                                                        'wygenerowania punktów '
                                                        'kompromisu '
                                                        '(trade-off) dla '
                                                        'progu.',
    'Adoption probability threshold': 'Próg prawdopodobieństwa dyskryminacji '
                                      'pozytywnej',
    'Precision': 'Precyzja predykcji (Precision)',
    'Recall': 'Czułość predykcji (Recall)',
    'F1': 'Metryka harmonijna (F1 Score)',
    'Flagged share': 'Frakcja obserwacji zakwalifikowanych ponad próg',
    'Threshold': 'Próg decyzyjny',
    'Metric value': 'Kwantyfikacja metryki',
    'Placement Risk Quadrant': 'Macierz identyfikacji przypadków o '
                               'podwyższonym ryzyku przedłużonego pobytu',
    'Predicted adoption probability': 'Estymowane prawdopodobieństwo sukcesu '
                                      'adopcyjnego',
    'Predicted days to outcome': 'Estymowany czas przetrzymywania do '
                                 'rozstrzygnięcia',
    'Error Slice Explorer': 'Dekompozycja strukturalna błędu estymatora',
    'H1: Intake vs Appearance': 'H1: Zmienne kontekstowe vs predyktory '
                                'fenotypowe',
    'H3: Age and Length of Stay': 'H3: Dystrybucja czasu pobytu w funkcji '
                                  'wieku',
    'H5: COVID-period Dynamics': 'H5: Oscylacje wolumenu zjawiska w okresie '
                                 'pandemii COVID-19',
    'Campaign Candidate Finder': 'Moduł selekcji kohortowej dla '
                                 'ukierunkowanych interwencji promocyjnych',
    'Exploratory cohort finder for groups that may benefit from targeted visibility. This is not causal recommendation logic.': 'Eksploracyjne '
                                                                                                                                'narzędzie '
                                                                                                                                'identyfikacji '
                                                                                                                                'podgrup '
                                                                                                                                'o '
                                                                                                                                'zaniżonym '
                                                                                                                                'profilu '
                                                                                                                                'adopcyjnym. '
                                                                                                                                'Wyselekcjonowane '
                                                                                                                                'podmioty '
                                                                                                                                'pełnią '
                                                                                                                                'rolę '
                                                                                                                                'propozycji '
                                                                                                                                'decyzyjnej '
                                                                                                                                'bez '
                                                                                                                                'domniemania '
                                                                                                                                'deterministycznej '
                                                                                                                                'sprawczości '
                                                                                                                                'kampanii.',
    'Run diagnostics to populate campaign cohorts.': 'Dokonaj ewaluacji '
                                                     'diagnostycznej, aby '
                                                     'wypełnić macierz '
                                                     'pre-wytypowanych grup '
                                                     'promocyjnych.',
    'All': 'Populacja całkowita',
    'No records match this cohort.': 'Nie zidentyfikowano wektorów '
                                     'spełniających warunki definicji wybranej '
                                     'kohorty.',
    'Cohort size': 'Liczebność (N) podgrupy',
    'Observed adoption': 'Wskaźnik adopcji - obserwacje empiryczne',
    'Mean predicted adoption': 'Wartość oczekiwana estymowanego '
                               'prawdopodobieństwa',
    'Median predicted days': 'Mediana prognozowanego czasu przetrzymywania '
                             '(dni)',
    'Campaign framing: this cohort may be useful for targeted visibility when predicted adoption probability is low or predicted days to outcome are high. Treat this as a prioritization signal, not proof of intervention impact.': 'Założenia '
                                                                                                                                                                                                                                      'operacyjne: '
                                                                                                                                                                                                                                      'wyszczególniona '
                                                                                                                                                                                                                                      'kohorta '
                                                                                                                                                                                                                                      'kwalifikuje '
                                                                                                                                                                                                                                      'się '
                                                                                                                                                                                                                                      'do '
                                                                                                                                                                                                                                      'objęcia '
                                                                                                                                                                                                                                      'zintensyfikowanym '
                                                                                                                                                                                                                                      'pakietem '
                                                                                                                                                                                                                                      'ekspozycji, '
                                                                                                                                                                                                                                      'ze '
                                                                                                                                                                                                                                      'względu '
                                                                                                                                                                                                                                      'na '
                                                                                                                                                                                                                                      'deficyt '
                                                                                                                                                                                                                                      'przewidywanego '
                                                                                                                                                                                                                                      'prawdopodobieństwa '
                                                                                                                                                                                                                                      'i/lub '
                                                                                                                                                                                                                                      'znacząco '
                                                                                                                                                                                                                                      'wydłużony '
                                                                                                                                                                                                                                      'prognozowany '
                                                                                                                                                                                                                                      'czas '
                                                                                                                                                                                                                                      'pobytu. '
                                                                                                                                                                                                                                      'Wynik '
                                                                                                                                                                                                                                      'ma '
                                                                                                                                                                                                                                      'charakter '
                                                                                                                                                                                                                                      'wspierający, '
                                                                                                                                                                                                                                      'a '
                                                                                                                                                                                                                                      'nie '
                                                                                                                                                                                                                                      'kauzalnie '
                                                                                                                                                                                                                                      'gwarantujący '
                                                                                                                                                                                                                                      'zwiększoną '
                                                                                                                                                                                                                                      'odnajdywalność '
                                                                                                                                                                                                                                      '(findability).',
    'Animal Type': 'Gatunek',
    'Age Group': 'Kategoria ontogenetyczna (Wiek)',
    'Intake Type': 'Charakterystyka trybu przyjęcia',
    'Covid Period': 'Dychotomia dla reżimu COVID-19',
    'Uses the combined CatBoost classifier and regressor when advanced artifacts exist. This is a demo prediction, not a causal decision rule.': 'Wykorzystuje '
                                                                                                                                                 'agregację '
                                                                                                                                                 'estymatorów '
                                                                                                                                                 '(klasyfikator '
                                                                                                                                                 'i '
                                                                                                                                                 'regresor '
                                                                                                                                                 'algorytmu '
                                                                                                                                                 'CatBoost). '
                                                                                                                                                 'Mechanizm '
                                                                                                                                                 'predykcji '
                                                                                                                                                 'odgrywa '
                                                                                                                                                 'rolę '
                                                                                                                                                 'demonstracyjną '
                                                                                                                                                 'i '
                                                                                                                                                 'nie '
                                                                                                                                                 'powinien '
                                                                                                                                                 'być '
                                                                                                                                                 'traktowany '
                                                                                                                                                 'jako '
                                                                                                                                                 'autonomiczna '
                                                                                                                                                 'funkcja '
                                                                                                                                                 'rozstrzygająca.',
    'Animal type': 'Gatunek',
    'Intake type': 'Klasyfikacja formalna trybu przyjęcia',
    'Intake condition': 'Stan kondycyjno-zdrowotny przy inkorporacji',
    'Sex upon intake': 'Status reprodukcyjny zwierzęcia w chwili inkorporacji',
    'Has name': 'Identyfikator nadany (Imię)',
    'Age in years': 'Zmienna ciągła wieku (lata)',
    'Breed': 'Taksonomia rasy',
    'Color': 'Klasyfikacja fenotypowa umaszczenia',
    'Intake date': 'Temporalny znacznik przyjęcia',
    'Run prediction': 'Zainicjuj predykcję heurystyczną',
    'Run `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv` first.': 'Czynność '
                                                                                                'uzależniona '
                                                                                                'od '
                                                                                                'wykonania '
                                                                                                'komendy '
                                                                                                'bazowej: '
                                                                                                '`python '
                                                                                                'scripts/train_advanced.py '
                                                                                                '--data '
                                                                                                'data/processed/modeling_dataset.csv`.',
    'Timeline group': 'Agregacja temporalna',
    'Run diagnostics to generate adoption timeline milestones.': 'Przetwórz '
                                                                 'zadania '
                                                                 'diagnostyczne '
                                                                 'by '
                                                                 'zdefiniować '
                                                                 'dystrybuantę '
                                                                 '(tzw. '
                                                                 'kamienie '
                                                                 'milowe) '
                                                                 'adopcji.',
    'Share adopted (%)': 'Poziom stopnia adopcji (%)',
    'Adoption timeline milestones': 'Wyznaczniki temporalne czasu do '
                                    'zwolnienia schroniskowego',
    'Generated Artifacts': 'Repozytorium wygenerowanych rezultatów '
                           'analitycznych',
    'Core commands:': 'Dyspozycje krytyczne (Core Commands):',
    'Reports directory:': 'Lokalizacja wyników tekstowych (Reports):',
    'Models directory:': 'Lokalizacja sieci oraz artefaktów modeli (Models):',
    'External Context Feature Test': 'Test predykcyjności skompresowanych '
                                     'zmiennych środowiskowych',
    'Run the commands below to populate `context_model_comparison.csv`.': 'Użyj '
                                                                          'zaprezentowanych '
                                                                          'komend '
                                                                          'dla '
                                                                          'zaludnienia '
                                                                          'pliku '
                                                                          '`context_model_comparison.csv`.',
    'Context features use intake-date weather and prior-window 311/intake-volume counts only.': 'Wymiar '
                                                                                                'zmiennych '
                                                                                                'kontekstowych '
                                                                                                'redukuje '
                                                                                                'się '
                                                                                                'do '
                                                                                                'logarytmicznie '
                                                                                                'przetworzonych '
                                                                                                'wektorów '
                                                                                                'wolumenowych, '
                                                                                                'raportów '
                                                                                                'interwencyjnych '
                                                                                                '(z '
                                                                                                'systemu '
                                                                                                '311) '
                                                                                                'oraz '
                                                                                                'profili '
                                                                                                'atmosferycznych '
                                                                                                'w '
                                                                                                'momentach '
                                                                                                'krańcowych '
                                                                                                'inkorporacji.',
    'Context minus base metric delta': 'Zmienność różniczkowa z zastosowaniem '
                                       'kontroli kontekstowych a podejściem '
                                       'zero-komponentowym (Base)',
    'Model': 'Paradygmat optymalizacyjny',
    'Effect': 'Wariancja wyniku',
    'Task': 'Zadana architektura (Task)',
    'improved': 'wzmocnienie predykcyjności',
    'worsened': 'degradacja jakości estymatora',
    'Dog': 'Pies',
    'Cat': 'Kot',
    'Stray': 'Obiekt porzucony/bezpański (Stray)',
    'Owner Surrender': 'Przekazanie na własność od dotychczasowego prawnego '
                       'opiekuna',
    'Public Assist': 'Pozyskanie w asyście służb państwowych',
    'Abandoned': 'Obiekt zarejestrowany jako porzucony',
    'Euthanasia Request': 'Protokół przyjęcia eutanazyjnego (Request)',
    'Normal': 'Normatywny (Normalny)',
    'Injured': 'Wybitnie urazowy (Ranny)',
    'Sick': 'Wybitnie chorobowy (Chory)',
    'Nursing': 'Status laktacyjny (Karmiący/a)',
    'Neonatal': 'Przedwcześnie odłączony/Noworodek (Neonatalny)',
    'Aged': 'Wybitnie podeszły wiek (Geriatryczny)',
    'Medical': 'O podwyższonych kryteriach asysty medycznej',
    'Behavior': 'O specyficznych ograniczeniach behawioralnych',
    'Other': 'Kategoria marginalizowana (Inna)',
    'Intact Male': 'Samiec niezdezintegrowany reprodukcyjnie',
    'Intact Female': 'Samica niezdezintegrowana reprodukcyjnie',
    'Neutered Male': 'Samiec pozbawiony zdolności reprodukcyjnych (Kastrat)',
    'Spayed Female': 'Samica pozbawiona zdolności reprodukcyjnych '
                     '(Wysterylizowana)',
    'Unknown': 'Kwantyfikator nierozpoznany (Unknown)',
    'unknown health': 'Aspekt zdrowotny nieoznaczony',
    'unknown behavior signal': 'Atrybut behawioralny niezakwalifikowany',
    ' Thesis Guide': 'Struktura manuskryptu oraz mapowanie referencji',
    'Model Sensitivity Demo': 'Moduł empirycznej ewaluacji podatności na '
                              'wejście (Sensitivity Analysis)',
    'Filter by Required for Thesis': 'Filtruj względem zbioru rezultatów '
                                     'obligatoryjnych',
    'Select Report to View': 'Dekompozycja jednostkowego sprawozdania '
                             'ewaluacyjnego',
    'Viewing Report:': 'Aktualna wizualizacja pliku:',
    'Report file not found on disk.': 'Zgłoszono brak fizycznego pliku '
                                      'artefaktu analitycznego.',
    'Target Definitions & Outcome Mappings': 'Aksjomatyczne definicje '
                                             'wskaźników celu a macierze '
                                             'alokacji rezultatów',
    'External Causal Validity & Generalisation Limits': 'Wiarygodność w sensie '
                                                        'zewnętrznym a '
                                                        'restrykcje '
                                                        'generalizacji '
                                                        '(Ograniczenia '
                                                        'wnioskowania '
                                                        'przyczynowo-skutkowego)',
    'Breed and Coat Colour Feature Engineering Justification': 'Prakseologiczne '
                                                               'uzasadnienie '
                                                               'konstrukcji '
                                                               'wyestrahowanych '
                                                               'atrybutów '
                                                               'morfologicznych '
                                                               '(rasa, maść)',
    'Machine Learning vs. Descriptive Non-ML Baselines': 'Dystynkcja rozwiązań '
                                                         'Machine Learning '
                                                         '(ML) na tle '
                                                         'algorytmów bez '
                                                         'modelu ukrytego '
                                                         '(Baseline)',
    'H1 — Intake Profile & Causal Context': 'H1 — Walidacja profilu '
                                            'okoliczności początkowych przy '
                                            'przyjęciu',
    'H2 — Seasonality & Intake Dynamics': 'H2 — Rozpoznanie fluktuacji w '
                                          'wymiarze sezonowym i strukturalnym '
                                          'wolumenu',
    'H3 — Age and Time-to-Outcome Timing': 'H3 — Kryterium ontogenetyczne a '
                                           'modelowanie estymaty czasu '
                                           'deinkorporacji (Time-to-Outcome)',
    'H4 — Coat Colour (Black/Dark Animal Syndrome Check)': 'H4 — Test '
                                                           'obciążenia '
                                                           'zjawiskiem tzw. '
                                                           "'Black/Dark Animal "
                                                           "Syndrome'",
    'H5 — COVID-Period Population Shift and Volume Impact': 'H5 — Wpływ '
                                                            'wstrząsu '
                                                            'pandemicznego na '
                                                            'strukturę '
                                                            'agregatów '
                                                            'ewidencji',
    'H1 - Intake Profile & Causal Context': 'H1 - Profil okoliczności '
                                            'początkowych przy przyjęciu',
    'H2 - Seasonality & Intake Dynamics': 'H2 - Fluktuacje w wymiarze '
                                          'sezonowym i strukturalnym',
    'H3 - Age and Time-to-Outcome Timing': 'H3 - Modelowanie estymaty czasu '
                                           'deinkorporacji',
    'H4 - Coat Colour (Black/Dark Animal Syndrome Check)': 'H4 - Test '
                                                           "'Black/Dark Animal "
                                                           "Syndrome'",
    'H5 - COVID-Period Population Shift and Volume Impact': 'H5 - Wpływ szoku '
                                                            'pandemicznego',
    'Hypothesis Evidence Matrix Summary': 'Wielowymiarowa macierz pokrycia '
                                          'weryfikacyjnego badanych hipotez',
    'Data Leakage Audit & Control Log': 'Ścisły dziennik ochrony spójności '
                                        'sekwencyjnej (Data Leakage Log)',
    'Propensity Score Matching Validation Examples': 'Dokumentacja estymatora '
                                                     'równoważności '
                                                     'prawdopodobieństw '
                                                     '(Propensity Score '
                                                     'Matching)',
    'Narrative Model Evidence & Key Findings Pack': 'Podstawowy wyciąg '
                                                    'sprawozdawczo-deskryptywny',
    'Subgroup Reliability & Underrepresented Cohorts': 'Rejestr parametrów '
                                                       'stabilności '
                                                       'subpopulacji '
                                                       'zjawiskowych i słabo '
                                                       'wyeksponowanych',
    'Final Model Architecture Selection': 'Argumentacja procesu selekcyjnego '
                                          'docelowej struktury algorytmicznej',
    'Optimal Classification Threshold & Utility Analysis': 'Matematyczna '
                                                           'wycena poziomu '
                                                           'kompromisu '
                                                           '(Optimal Decision '
                                                           'Boundary)',
    'Model Probability Calibration Interpretation': 'Przełożenie fizyczne '
                                                    'skategoryzowanych '
                                                    'rozkładów wiarygodności '
                                                    'modelu',
    'Operational Risk & Model Reliability Red Flags': 'Metryki zagrożenia '
                                                      'stabilności wykonawczej '
                                                      'z odnotowaniem tzw. Red '
                                                      'Flags',
    'Data Pipeline Audit & Attrition Logging': 'Log procesów potoku '
                                               'analitycznego włączając '
                                               'odsetek utraty informacji na '
                                               'stadiach konwersji (Attrition '
                                               'Logging)',
    'Reproducibility Snapshot & Environment Info': 'Wektor ewaluacji '
                                                   'stabilności i spójności '
                                                   'konfiguracyjnej ekosystemu',
    'quick placement likely': 'przewidywana bezzwłoczna relokacja docelowa',
    'needs visibility': 'konieczność amplifikacji ekspozycji (Promocja)',
    'long-stay risk': 'obciążony znacznym stopniem ryzyka zamrożenia stanu',
    'outcome support priority': 'sygnalizowany tryb wsparcia pozaobszarowego '
                                '(Priorytet decyzyjny)',
    'Classification PR-AUC': 'Obszar pod krzywą PR (Precision-Recall AUC) na '
                             'modelu klasyfikującym',
    'Classification ROC-AUC': 'Obszar pod krzywą ROC (Receiver Operating '
                              'Characteristic) dla klasyfikacji',
    'Classification F1': 'Wartość harmoniczna błędu uśrednionego (F1-score)',
    'Regression MAE': 'Średni moduł błędu absolutnego (MAE) dla regresji',
    'Regression RMSE': 'Pierwiastek błędu średniokwadratowego (RMSE)',
    'Predicted adoption chance (calibrated)': 'Zrekalibrowana estymata '
                                              'prawdopodobieństwa sukcesu '
                                              'adopcyjnego',
    'Predicted adoption probability (calibrated)': 'Ujednolicone kalibracyjnie '
                                                   'prawdopodobieństwo '
                                                   'dyskryminacji',
    '**Adoption Rates by Intake Type**': '**Ewaluacja stopnia sprawności '
                                         'procesu adopcyjnego w warunkach '
                                         'klasyfikacji startowej**',
    '**Finding:** Older animals face significant penalties in adoption likelihood. Wait times to any outcome are complex, as seniors may leave the shelter faster due to higher rates of non-adoption outcomes.': '**Postulat '
                                                                                                                                                                                                                  'dedukcyjny:** '
                                                                                                                                                                                                                  'Znaczący '
                                                                                                                                                                                                                  'spadek '
                                                                                                                                                                                                                  'współczynników '
                                                                                                                                                                                                                  'weryfikujących '
                                                                                                                                                                                                                  'pozytywne '
                                                                                                                                                                                                                  'rezultaty '
                                                                                                                                                                                                                  'adopcyjne '
                                                                                                                                                                                                                  'ujawniono '
                                                                                                                                                                                                                  'w '
                                                                                                                                                                                                                  'odniesieniu '
                                                                                                                                                                                                                  'do '
                                                                                                                                                                                                                  'podzbioru '
                                                                                                                                                                                                                  'osobników '
                                                                                                                                                                                                                  'starzejących '
                                                                                                                                                                                                                  'się. '
                                                                                                                                                                                                                  'Parametr '
                                                                                                                                                                                                                  'dystrybucji '
                                                                                                                                                                                                                  'czasowej '
                                                                                                                                                                                                                  'charakteryzuje '
                                                                                                                                                                                                                  'się '
                                                                                                                                                                                                                  'tu '
                                                                                                                                                                                                                  'podwyższonym '
                                                                                                                                                                                                                  'wskaźnikiem '
                                                                                                                                                                                                                  'skomplikowania, '
                                                                                                                                                                                                                  'jako '
                                                                                                                                                                                                                  'następstwo '
                                                                                                                                                                                                                  'zwiększonego '
                                                                                                                                                                                                                  'tempa '
                                                                                                                                                                                                                  'deinkorporacji '
                                                                                                                                                                                                                  'z '
                                                                                                                                                                                                                  'zastosowaniem '
                                                                                                                                                                                                                  'rozwiązań '
                                                                                                                                                                                                                  'nieadopcyjnych, '
                                                                                                                                                                                                                  'takich '
                                                                                                                                                                                                                  'jak '
                                                                                                                                                                                                                  'eutanazja '
                                                                                                                                                                                                                  'lub '
                                                                                                                                                                                                                  'specjalistyczna '
                                                                                                                                                                                                                  'opieka '
                                                                                                                                                                                                                  'terminalna.',
    '**Finding:** Physical appearance (Breed and Color) and Age are the strongest predictors of adoption, significantly outweighing the context of how an animal arrives (Intake Circumstances and Condition).': '**Postulat '
                                                                                                                                                                                                                 'dedukcyjny:** '
                                                                                                                                                                                                                 'Elementarne '
                                                                                                                                                                                                                 'parametry '
                                                                                                                                                                                                                 'predykcji '
                                                                                                                                                                                                                 'fenotypowej '
                                                                                                                                                                                                                 '(tj. '
                                                                                                                                                                                                                 'kategoryzacja '
                                                                                                                                                                                                                 'rasy '
                                                                                                                                                                                                                 'i '
                                                                                                                                                                                                                 'umaszczenie) '
                                                                                                                                                                                                                 'deklasują '
                                                                                                                                                                                                                 'czynniki '
                                                                                                                                                                                                                 'związane '
                                                                                                                                                                                                                 'stricte '
                                                                                                                                                                                                                 'z '
                                                                                                                                                                                                                 'tłem '
                                                                                                                                                                                                                 'incydentu '
                                                                                                                                                                                                                 'prowadzącego '
                                                                                                                                                                                                                 'do '
                                                                                                                                                                                                                 'przyjęcia '
                                                                                                                                                                                                                 'schroniskowego '
                                                                                                                                                                                                                 '(tzw. '
                                                                                                                                                                                                                 'kody '
                                                                                                                                                                                                                 'klasyfikacyjne '
                                                                                                                                                                                                                 'zdarzeń). '
                                                                                                                                                                                                                 'W '
                                                                                                                                                                                                                 'połączeniu '
                                                                                                                                                                                                                 'z '
                                                                                                                                                                                                                 'wiekiem, '
                                                                                                                                                                                                                 'dominują '
                                                                                                                                                                                                                 'jako '
                                                                                                                                                                                                                 'wiodący '
                                                                                                                                                                                                                 'predyktor '
                                                                                                                                                                                                                 'rezultatu '
                                                                                                                                                                                                                 'wyjściowego.',
    '**Finding:** The COVID-19 pandemic period was associated with a marked increase in adoption rates and a reduction in total volume. These period shifts must be accounted for to prevent model drift.': '**Postulat '
                                                                                                                                                                                                            'dedukcyjny:** '
                                                                                                                                                                                                            'Wyznaczony '
                                                                                                                                                                                                            'kwantyl '
                                                                                                                                                                                                            'temporalny '
                                                                                                                                                                                                            'dla '
                                                                                                                                                                                                            'zaistnienia '
                                                                                                                                                                                                            'pandemii '
                                                                                                                                                                                                            'COVID-19, '
                                                                                                                                                                                                            'wskazał '
                                                                                                                                                                                                            'wyraźny '
                                                                                                                                                                                                            'wzrost '
                                                                                                                                                                                                            'absorpcji '
                                                                                                                                                                                                            'zjawisk '
                                                                                                                                                                                                            'adopcyjnych, '
                                                                                                                                                                                                            'pomimo '
                                                                                                                                                                                                            'regresji '
                                                                                                                                                                                                            'całkowitego '
                                                                                                                                                                                                            'wolumenu '
                                                                                                                                                                                                            'zarejestrowanych '
                                                                                                                                                                                                            'inkorporacji. '
                                                                                                                                                                                                            'Te '
                                                                                                                                                                                                            'dewiacje '
                                                                                                                                                                                                            'strukturalne '
                                                                                                                                                                                                            'populacji '
                                                                                                                                                                                                            'wymagają '
                                                                                                                                                                                                            'starannej '
                                                                                                                                                                                                            'mitygacji '
                                                                                                                                                                                                            'w '
                                                                                                                                                                                                            'estymatorze '
                                                                                                                                                                                                            'w '
                                                                                                                                                                                                            'trosce '
                                                                                                                                                                                                            'o '
                                                                                                                                                                                                            'rygor '
                                                                                                                                                                                                            'uniknięcia '
                                                                                                                                                                                                            'zjawiska '
                                                                                                                                                                                                            'concept '
                                                                                                                                                                                                            'drift '
                                                                                                                                                                                                            '(odchylenia '
                                                                                                                                                                                                            'pojęciowego '
                                                                                                                                                                                                            'modelu).',
    '**Global SHAP Feature Importance (Classification)**': '**Wyodrębnienie '
                                                           'kluczowej '
                                                           'hierarchii wpływów '
                                                           'parametrów według '
                                                           'dystrybucji w '
                                                           'algorytmach SHAP '
                                                           '(klasyfikator)**',
    '**Median Days to Outcome by Age**': '**Wartości mediany dla czasu '
                                         'ekspozycji wyznaczane wymiarem '
                                         'kategorii wiekowej (Age)**',
    '**Volume and Outcomes Across Periods**': '**Przekrojowa miara natężenia '
                                              'operacji wejściowych i '
                                              'deinkorporacji rozpatrywana na '
                                              'płaszczyznach historycznych**',
    'All generated thesis deliverables, target definitions, and validation reports are listed below.': 'Dokument '
                                                                                                       'stanowi '
                                                                                                       'skondensowany '
                                                                                                       'indeks '
                                                                                                       'referencji '
                                                                                                       'powołujących '
                                                                                                       'się '
                                                                                                       'na '
                                                                                                       'wszystkie '
                                                                                                       'dedykowane '
                                                                                                       'zasoby '
                                                                                                       'badawcze '
                                                                                                       '(dossier) '
                                                                                                       'wytworzone '
                                                                                                       'przez '
                                                                                                       'pipeline '
                                                                                                       'analityczny '
                                                                                                       'włącznie '
                                                                                                       'z '
                                                                                                       'parametrami '
                                                                                                       'walidacyjnymi '
                                                                                                       'i '
                                                                                                       'konkluzjami '
                                                                                                       'testów '
                                                                                                       'empirycznych.',
    'Artifact Manifest': 'Inwentarz Artefaktów Bazy',
    'Artifact Path': 'Wyznaczona domena i ścieżka dostępu',
    'Artifact Type': 'Rodzaj zarejestrowanego zasobu',
    'Automated euthanasia triaging': 'Skryptowa automatyzacja decyzji '
                                     'eutanazyjnych (Triage)',
    'Best Use Case': 'Wektor racjonalnej użyteczności (Best Practice)',
    'CatBoost consistently outperformed baseline models at separating adoptions from other outcomes, offering the highest precision-recall area under the curve (PR-AUC) for this imbalanced task.': 'Zaimplementowana '
                                                                                                                                                                                                     'metoda '
                                                                                                                                                                                                     'klasyfikacji '
                                                                                                                                                                                                     'CatBoost '
                                                                                                                                                                                                     'konsekwentnie '
                                                                                                                                                                                                     'neutralizowała '
                                                                                                                                                                                                     'asymetryczną '
                                                                                                                                                                                                     'skuteczność '
                                                                                                                                                                                                     'modeli '
                                                                                                                                                                                                     'wyjściowych, '
                                                                                                                                                                                                     'dyskryminując '
                                                                                                                                                                                                     'w '
                                                                                                                                                                                                     'sposób '
                                                                                                                                                                                                     'najwłaściwszy '
                                                                                                                                                                                                     'procesy '
                                                                                                                                                                                                     'adopcyjne '
                                                                                                                                                                                                     'i '
                                                                                                                                                                                                     'generując '
                                                                                                                                                                                                     'bezkonkurencyjne '
                                                                                                                                                                                                     'pole '
                                                                                                                                                                                                     'dla '
                                                                                                                                                                                                     'ewaluacji '
                                                                                                                                                                                                     'metodą '
                                                                                                                                                                                                     'PR-AUC '
                                                                                                                                                                                                     'na '
                                                                                                                                                                                                     'mocno '
                                                                                                                                                                                                     'niezbilansowanym '
                                                                                                                                                                                                     'zbiorze '
                                                                                                                                                                                                     'docelowym.',
    'Chapter': 'Pozycja Rozdziału',
    'Data only reflects intake time': 'Parametryzacją wejściową objęto '
                                      'wyłącznie stany zarejestrowane '
                                      'adekwatnie z logiem czasowym początku '
                                      "obserwacji. Model ewaluacyjnie 'nie "
                                      "zna' późniejszych uwarunkowań "
                                      'adaptacyjnych osobnika w schronisku.',
    'Days': 'Okres kwantyfikowany w domenie jednostek dziennych',
    'Exists': 'Rozpoznany fizycznie',
    'For predicting the exact length of stay, CatBoost provided the lowest average error. However, length-of-stay is highly skewed and right-censored.': 'Dla '
                                                                                                                                                         'optymalizacji '
                                                                                                                                                         'i '
                                                                                                                                                         'predykcji '
                                                                                                                                                         'surowej '
                                                                                                                                                         'zmiennej '
                                                                                                                                                         'ciągłej '
                                                                                                                                                         'reprezentującej '
                                                                                                                                                         'wymiar '
                                                                                                                                                         'czasowy '
                                                                                                                                                         'retencji, '
                                                                                                                                                         'podejście '
                                                                                                                                                         'estymacyjne '
                                                                                                                                                         'na '
                                                                                                                                                         'rdzeniu '
                                                                                                                                                         'CatBoost '
                                                                                                                                                         'wygenerowało '
                                                                                                                                                         'najniższy '
                                                                                                                                                         'współczynnik '
                                                                                                                                                         'MAE. '
                                                                                                                                                         'Koniecznym '
                                                                                                                                                         'jest '
                                                                                                                                                         'zaobsersem, '
                                                                                                                                                         'że '
                                                                                                                                                         'w '
                                                                                                                                                         'ujęciu '
                                                                                                                                                         'holistycznym '
                                                                                                                                                         'zmienna '
                                                                                                                                                         'ta '
                                                                                                                                                         'ma '
                                                                                                                                                         'mocno '
                                                                                                                                                         'prawostronną '
                                                                                                                                                         'deformację '
                                                                                                                                                         'i '
                                                                                                                                                         'ulega '
                                                                                                                                                         'procesowi '
                                                                                                                                                         'cenzurowania '
                                                                                                                                                         'z '
                                                                                                                                                         'tejże '
                                                                                                                                                         'strony.',
    'Higher is better. Evaluated on out-of-time test set.': 'Stymulacja '
                                                            'optymalizacyjna '
                                                            'przez '
                                                            'maksymalizację. '
                                                            'Badanie i '
                                                            'ostateczna '
                                                            'walidacja '
                                                            'dokonana na '
                                                            'chronologicznym '
                                                            'kwantylu '
                                                            'walidacyjnym '
                                                            '(Out-of-time test '
                                                            'set).',
    'Importance Impact': 'Absolutny mnożnik korelacji i wpływu (Impact '
                         'Magnitude)',
    'Lower is better. Mean absolute error on test set.': 'Stymulacja '
                                                         'optymalizacyjna '
                                                         'przez minimalizację '
                                                         'dysonansu (Mean '
                                                         'Absolute Error). '
                                                         'Badania zatwierdzone '
                                                         'na kwantylu '
                                                         'walidacyjnym w '
                                                         'warunkach odciętej '
                                                         'ewidencji.',
    'Model output is unavailable. Check model artifacts and metadata.': 'Wyjście '
                                                                        'predykcyjne '
                                                                        'dla '
                                                                        'zadanych '
                                                                        'zapytań '
                                                                        'uległo '
                                                                        'zatrzymaniu '
                                                                        'brakiem '
                                                                        'danych. '
                                                                        'Zbadaj '
                                                                        'wektory '
                                                                        'artefaktów '
                                                                        'predyktora '
                                                                        'oraz '
                                                                        'jego '
                                                                        'metadane.',
    'No classification artifacts found.': 'Dysk nie identyfikuje wyników '
                                          'estymacji dla problemu '
                                          'klasyfikacji.',
    'No model is perfect. Here is exactly where the model struggles and the magnitude of its errors:': 'Ułomność '
                                                                                                       'strukturalna '
                                                                                                       'każdego '
                                                                                                       'systemu '
                                                                                                       'analitycznego '
                                                                                                       'wymusza '
                                                                                                       'obnażenie '
                                                                                                       'sektorów '
                                                                                                       'usterkowości. '
                                                                                                       'Oto '
                                                                                                       'dokładne '
                                                                                                       'zobrazowanie '
                                                                                                       'grup, '
                                                                                                       'w '
                                                                                                       'których '
                                                                                                       'estymator '
                                                                                                       'wykazuje '
                                                                                                       'wysoki '
                                                                                                       'stopień '
                                                                                                       'deficytu '
                                                                                                       'precyzji '
                                                                                                       'wraz '
                                                                                                       'ze '
                                                                                                       'skalowaniem '
                                                                                                       'rozmiaru '
                                                                                                       'dewiacji:',
    'No regression artifacts found.': 'Dysk nie identyfikuje logów błędów ani '
                                      'metryk ewaluacyjnych modelu w trybie '
                                      'ciągłym (Regresji).',
    'Notes': 'Obserwacje konkludujące',
    'Prediction failed.': 'Proces propagacji dla wektora zapytań wejściowych '
                          'zawiódł na warstwie błędu krytycznego.',
    'Primary Limitation': 'Podstawowy element dymisji obiektywności '
                          '(Limitation Limit)',
    'Primary Metric (MAE)': 'Docelowy Wskaźnik Weryfikacji Optymalizacyjnej '
                            '(MAE - Błąd Bezwzględny Zmiennej Ciągłej)',
    'Primary Metric (PR-AUC)': 'Docelowy Wskaźnik Weryfikacji Optymalizacyjnej '
                               '(PR-AUC - Czułość vs Precyzja względem '
                               'Prawdopodobieństwa Klasyfikacyjnego)',
    'Prioritizing visibility campaigns': 'Mechanizm asocjacyjno-heurystyczny '
                                         'dla dyskryminacji celów kampanii '
                                         'podbijających odnajdywalność '
                                         'społecznościową (Visibility '
                                         'Profiling)',
    'Read Thesis & Methodology Reports': 'Odszukaj odnośnik do manuskryptu '
                                         'wraz z argumentacjami '
                                         'metodologicznymi i aparatami użytych '
                                         'pojęć.',
    'Refresh Data': 'Resynchronizuj Bufory Pamięci Stanu (Refresh)',
    'Required': 'Wyróżnik Krytyczności dla Wektorów Raportów Dyplomowych',
    'Riskiest Use Case': 'Błędy poznawcze w implementacjach wysokiego szczebla '
                         'krytycznego',
    'Run `python scripts/generate_diagnostics.py` to see error slices.': 'Wykonaj '
                                                                         'skryptowy '
                                                                         'mechanizm '
                                                                         '`python '
                                                                         'scripts/generate_diagnostics.py` '
                                                                         'by '
                                                                         'zainicjować '
                                                                         'agregacje '
                                                                         'w '
                                                                         'rozbiciu '
                                                                         'na '
                                                                         'rzuty '
                                                                         'błędu '
                                                                         'predykcji.',
    'Run `python scripts/generate_evidence_pack.py` to see failure modes.': 'Ewaluuj '
                                                                            'błędy '
                                                                            'z '
                                                                            'dekompozycją '
                                                                            'wykonując '
                                                                            'polecenie '
                                                                            '`python '
                                                                            'scripts/generate_evidence_pack.py` '
                                                                            'w '
                                                                            'konsoli '
                                                                            'uruchomieniowej.',
    'Source Script': 'Mechanizm Wytwórczy (Źródłowy Kontroler Rutynowy)',
    'The machine learning pipeline evaluated logistic regression, random forests, histogram gradient boosting, and CatBoost models. Here is the final selection based on empirical validation data:': 'Architektura '
                                                                                                                                                                                                      'badawcza '
                                                                                                                                                                                                      'systemu '
                                                                                                                                                                                                      'przeegzaminowała '
                                                                                                                                                                                                      'estymatory '
                                                                                                                                                                                                      'Regresji '
                                                                                                                                                                                                      'Logistycznej, '
                                                                                                                                                                                                      'Lasów '
                                                                                                                                                                                                      'Losowych, '
                                                                                                                                                                                                      'estymacje '
                                                                                                                                                                                                      'zoptymalizowane '
                                                                                                                                                                                                      'Histogramami '
                                                                                                                                                                                                      'Wzmacniania '
                                                                                                                                                                                                      'Gradientowego '
                                                                                                                                                                                                      'po '
                                                                                                                                                                                                      'implementacje '
                                                                                                                                                                                                      'wyższych '
                                                                                                                                                                                                      'rzędów '
                                                                                                                                                                                                      'optymalizacyjnych '
                                                                                                                                                                                                      'drzew '
                                                                                                                                                                                                      'kategorycznych, '
                                                                                                                                                                                                      'jakimi '
                                                                                                                                                                                                      'są '
                                                                                                                                                                                                      'mechanizmy '
                                                                                                                                                                                                      'rzędu '
                                                                                                                                                                                                      'CatBoost. '
                                                                                                                                                                                                      'Ustalenia '
                                                                                                                                                                                                      'zaprezentowane '
                                                                                                                                                                                                      'poniżej, '
                                                                                                                                                                                                      'dokonane '
                                                                                                                                                                                                      'są '
                                                                                                                                                                                                      'wyłącznie '
                                                                                                                                                                                                      'wehikułem '
                                                                                                                                                                                                      'wyodrębnionych '
                                                                                                                                                                                                      'procesów '
                                                                                                                                                                                                      'podzbiorów '
                                                                                                                                                                                                      'obarczonych '
                                                                                                                                                                                                      'atrybutem '
                                                                                                                                                                                                      'prawdy '
                                                                                                                                                                                                      'empirycznej '
                                                                                                                                                                                                      'z '
                                                                                                                                                                                                      'okresu '
                                                                                                                                                                                                      'zamkniętego:',
    "The regression model's Mean Absolute Error (MAE) varies drastically by subgroup:": 'Miara '
                                                                                        'uśrednionego '
                                                                                        'błędu '
                                                                                        'w '
                                                                                        'trybie '
                                                                                        'estymacji '
                                                                                        'długości '
                                                                                        'oczekiwania '
                                                                                        'ma '
                                                                                        'tendencję '
                                                                                        'do '
                                                                                        'drastycznej '
                                                                                        'wariacji '
                                                                                        'zależnie '
                                                                                        'od '
                                                                                        'identyfikacji '
                                                                                        'kohorty:',
    'This dashboard translates raw shelter data into concrete evidence. Below are the finalized findings across the primary thesis hypotheses.': 'Platforma '
                                                                                                                                                 'udostępniona '
                                                                                                                                                 'w '
                                                                                                                                                 'tej '
                                                                                                                                                 'przestrzeni '
                                                                                                                                                 'pełni '
                                                                                                                                                 'funkcję '
                                                                                                                                                 'agregatora '
                                                                                                                                                 'wnioskującego '
                                                                                                                                                 'z '
                                                                                                                                                 'wymiaru '
                                                                                                                                                 'surowych '
                                                                                                                                                 'danych '
                                                                                                                                                 'ewidencyjnych '
                                                                                                                                                 'po '
                                                                                                                                                 'postulat '
                                                                                                                                                 'dowodowy '
                                                                                                                                                 'w '
                                                                                                                                                 'ujęciu '
                                                                                                                                                 'akademickim. '
                                                                                                                                                 'Bezpośrednio '
                                                                                                                                                 'pod '
                                                                                                                                                 'komunikatem '
                                                                                                                                                 'osadzono '
                                                                                                                                                 'ostateczne '
                                                                                                                                                 'ramy '
                                                                                                                                                 'dowodowe '
                                                                                                                                                 'odnoszące '
                                                                                                                                                 'się '
                                                                                                                                                 'do '
                                                                                                                                                 'priorytetowych '
                                                                                                                                                 'postulatów '
                                                                                                                                                 'hipotez '
                                                                                                                                                 'badawczych '
                                                                                                                                                 'sformułowanych '
                                                                                                                                                 'w '
                                                                                                                                                 'obrębie '
                                                                                                                                                 'dysertacji:',
    'Uses the combined classifier and regressor when advanced artifacts exist. This is a demo prediction, not a causal decision rule.': 'Rutyna '
                                                                                                                                        'wymusza '
                                                                                                                                        'pobranie '
                                                                                                                                        'zasobów '
                                                                                                                                        'z '
                                                                                                                                        'potoku '
                                                                                                                                        'wytworzonego '
                                                                                                                                        'predyktora '
                                                                                                                                        '(skonsolidowany '
                                                                                                                                        'algorytm '
                                                                                                                                        'klasyfikująco-regresyjny). '
                                                                                                                                        'Narzędzie '
                                                                                                                                        'to '
                                                                                                                                        'stanowi '
                                                                                                                                        'symulację '
                                                                                                                                        'estymacyjną '
                                                                                                                                        'oraz '
                                                                                                                                        'weryfikację '
                                                                                                                                        'na '
                                                                                                                                        'żądanie '
                                                                                                                                        'i '
                                                                                                                                        'w '
                                                                                                                                        'żadnym '
                                                                                                                                        'ze '
                                                                                                                                        'swoich '
                                                                                                                                        'obiektywnych '
                                                                                                                                        'wskazań '
                                                                                                                                        'nie '
                                                                                                                                        'uprawnia '
                                                                                                                                        'do '
                                                                                                                                        'stanowienia '
                                                                                                                                        'niezależnej '
                                                                                                                                        'i '
                                                                                                                                        'odciętej '
                                                                                                                                        'reguły '
                                                                                                                                        'przyzwolenia '
                                                                                                                                        'na '
                                                                                                                                        'określoną '
                                                                                                                                        'decyzję '
                                                                                                                                        'fizyczną '
                                                                                                                                        'względem '
                                                                                                                                        'zjawiska.',
    'View Full Executive Summary Report': 'Rozwiń dekompozycję całościowego '
                                          'ustępstwa wykonawczego dla '
                                          'streszczenia dysertacji',
    'When the model misclassifies an outcome, these are the most common failure modes:': 'Dezintegracja '
                                                                                         'struktury '
                                                                                         'błędu '
                                                                                         'i '
                                                                                         'odchyleń '
                                                                                         'predyktora. '
                                                                                         'Pod '
                                                                                         'spodem '
                                                                                         'osadzono '
                                                                                         'klasyfikatory '
                                                                                         'najczęstszych '
                                                                                         'dysonansów '
                                                                                         'i '
                                                                                         'omyłek '
                                                                                         'estymatora '
                                                                                         'w '
                                                                                         'stosunku '
                                                                                         'do '
                                                                                         'obserwowalnej '
                                                                                         'w '
                                                                                         'przyszłości '
                                                                                         'materii '
                                                                                         'zdarzeniowej:',
    'While machine learning successfully ranks animals by placement difficulty, **these predictions are associative, not causal.**': 'Konstrukty '
                                                                                                                                     'probabilistyczne '
                                                                                                                                     'poddawane '
                                                                                                                                     'treningowi '
                                                                                                                                     'na '
                                                                                                                                     'surowych '
                                                                                                                                     'danych '
                                                                                                                                     'statystycznych '
                                                                                                                                     'potwierdzają '
                                                                                                                                     'się '
                                                                                                                                     'w '
                                                                                                                                     'budowaniu '
                                                                                                                                     'wektorów '
                                                                                                                                     'rangujących, '
                                                                                                                                     'w '
                                                                                                                                     'aspekcie '
                                                                                                                                     'priorytetyzacji '
                                                                                                                                     'obciążeń '
                                                                                                                                     'adopcyjnych '
                                                                                                                                     'na '
                                                                                                                                     'zbiorach '
                                                                                                                                     'podmiotów. '
                                                                                                                                     'Pomimo '
                                                                                                                                     'tych '
                                                                                                                                     'możliwości '
                                                                                                                                     '**prezentowane '
                                                                                                                                     'ujęcia '
                                                                                                                                     'estymacyjne '
                                                                                                                                     'w '
                                                                                                                                     'całości '
                                                                                                                                     'obarczone '
                                                                                                                                     'są '
                                                                                                                                     'asocjacją '
                                                                                                                                     'cech, '
                                                                                                                                     'całkowicie '
                                                                                                                                     'wykluczając '
                                                                                                                                     'bezpośrednie '
                                                                                                                                     'domniemania '
                                                                                                                                     'kauzalne '
                                                                                                                                     'pomiędzy '
                                                                                                                                     'atrybutem '
                                                                                                                                     'podmiotu '
                                                                                                                                     'a '
                                                                                                                                     'generowaną '
                                                                                                                                     'w '
                                                                                                                                     'czasie '
                                                                                                                                     'przyszłym '
                                                                                                                                     'reakcją '
                                                                                                                                     'środowiska.**',
    "While puppies and kittens ('baby') often leave the shelter within 6-7 days, 'adult' animals face similar wait times. 'Senior' animals exhibit shorter median days to any outcome (e.g. 4.2 days), reflecting alternative outcome pathways. Age remains a critical predictive feature.": 'Pomimo '
                                                                                                                                                                                                                                                                                             'ewidentnego '
                                                                                                                                                                                                                                                                                             'odchylenia '
                                                                                                                                                                                                                                                                                             'obserwowalnego '
                                                                                                                                                                                                                                                                                             'czasu '
                                                                                                                                                                                                                                                                                             'opuszczenia '
                                                                                                                                                                                                                                                                                             'stanowisk '
                                                                                                                                                                                                                                                                                             'ewidencji '
                                                                                                                                                                                                                                                                                             'przez '
                                                                                                                                                                                                                                                                                             'grupę '
                                                                                                                                                                                                                                                                                             'noworodkową '
                                                                                                                                                                                                                                                                                             'rzędu '
                                                                                                                                                                                                                                                                                             'do '
                                                                                                                                                                                                                                                                                             '7 '
                                                                                                                                                                                                                                                                                             'jednostek '
                                                                                                                                                                                                                                                                                             '(Dni), '
                                                                                                                                                                                                                                                                                             'badana '
                                                                                                                                                                                                                                                                                             'waga '
                                                                                                                                                                                                                                                                                             "'wielkości "
                                                                                                                                                                                                                                                                                             "dorosłych' "
                                                                                                                                                                                                                                                                                             'generuje '
                                                                                                                                                                                                                                                                                             'wariancję '
                                                                                                                                                                                                                                                                                             'czasową '
                                                                                                                                                                                                                                                                                             'bardzo '
                                                                                                                                                                                                                                                                                             'zbliżoną '
                                                                                                                                                                                                                                                                                             'do '
                                                                                                                                                                                                                                                                                             'tych '
                                                                                                                                                                                                                                                                                             'pierwszych '
                                                                                                                                                                                                                                                                                             'grup '
                                                                                                                                                                                                                                                                                             'wiekowych. '
                                                                                                                                                                                                                                                                                             'Najciekawsze '
                                                                                                                                                                                                                                                                                             'odchylenia '
                                                                                                                                                                                                                                                                                             'struktury '
                                                                                                                                                                                                                                                                                             'modelu '
                                                                                                                                                                                                                                                                                             'leżą '
                                                                                                                                                                                                                                                                                             'w '
                                                                                                                                                                                                                                                                                             'rzędzie '
                                                                                                                                                                                                                                                                                             'kwantyli '
                                                                                                                                                                                                                                                                                             'odzwierciedlonych '
                                                                                                                                                                                                                                                                                             'wartością '
                                                                                                                                                                                                                                                                                             'skróconą '
                                                                                                                                                                                                                                                                                             'dla '
                                                                                                                                                                                                                                                                                             'osobnika '
                                                                                                                                                                                                                                                                                             'powyżej '
                                                                                                                                                                                                                                                                                             'tzw. '
                                                                                                                                                                                                                                                                                             'wieku '
                                                                                                                                                                                                                                                                                             'geriatrycznego '
                                                                                                                                                                                                                                                                                             "'Senior'. "
                                                                                                                                                                                                                                                                                             'Czas '
                                                                                                                                                                                                                                                                                             'opuszczenia '
                                                                                                                                                                                                                                                                                             'murów '
                                                                                                                                                                                                                                                                                             'instytucji '
                                                                                                                                                                                                                                                                                             'bywa '
                                                                                                                                                                                                                                                                                             'tu '
                                                                                                                                                                                                                                                                                             'niższy '
                                                                                                                                                                                                                                                                                             'od '
                                                                                                                                                                                                                                                                                             'standardowego '
                                                                                                                                                                                                                                                                                             '(ok. '
                                                                                                                                                                                                                                                                                             '4 '
                                                                                                                                                                                                                                                                                             'dni). '
                                                                                                                                                                                                                                                                                             'Dyskrepancja '
                                                                                                                                                                                                                                                                                             'jest '
                                                                                                                                                                                                                                                                                             'powodowana '
                                                                                                                                                                                                                                                                                             'ucieczką '
                                                                                                                                                                                                                                                                                             'tych '
                                                                                                                                                                                                                                                                                             'podmiotów '
                                                                                                                                                                                                                                                                                             'od '
                                                                                                                                                                                                                                                                                             'założeń '
                                                                                                                                                                                                                                                                                             'procesu '
                                                                                                                                                                                                                                                                                             'klasycznej '
                                                                                                                                                                                                                                                                                             'opieki '
                                                                                                                                                                                                                                                                                             'docelowej, '
                                                                                                                                                                                                                                                                                             'co '
                                                                                                                                                                                                                                                                                             'często '
                                                                                                                                                                                                                                                                                             'zamyka '
                                                                                                                                                                                                                                                                                             'te '
                                                                                                                                                                                                                                                                                             'trajektorie '
                                                                                                                                                                                                                                                                                             'wynikiem '
                                                                                                                                                                                                                                                                                             "'eutanazji' "
                                                                                                                                                                                                                                                                                             'czy '
                                                                                                                                                                                                                                                                                             'przymusu '
                                                                                                                                                                                                                                                                                             'objęcia '
                                                                                                                                                                                                                                                                                             'reżimową '
                                                                                                                                                                                                                                                                                             'farmakoterapią '
                                                                                                                                                                                                                                                                                             'poza '
                                                                                                                                                                                                                                                                                             'ośrodkiem '
                                                                                                                                                                                                                                                                                             'schroniskowym. '
                                                                                                                                                                                                                                                                                             'Kryterium '
                                                                                                                                                                                                                                                                                             'ontogenetyczne '
                                                                                                                                                                                                                                                                                             'jest '
                                                                                                                                                                                                                                                                                             'tym '
                                                                                                                                                                                                                                                                                             'samym '
                                                                                                                                                                                                                                                                                             'uznane '
                                                                                                                                                                                                                                                                                             'za '
                                                                                                                                                                                                                                                                                             'fundamentalną '
                                                                                                                                                                                                                                                                                             'właściwość '
                                                                                                                                                                                                                                                                                             'decyzyjno-różniczkową '
                                                                                                                                                                                                                                                                                             'we '
                                                                                                                                                                                                                                                                                             'wszystkich '
                                                                                                                                                                                                                                                                                             'architekturach '
                                                                                                                                                                                                                                                                                             'docelowych '
                                                                                                                                                                                                                                                                                             'budowanych '
                                                                                                                                                                                                                                                                                             'modeli.',
    ' Interpretation limits': ' Restrykcje poznawcze logiki asocjacyjnej '
                                'dla wymiarów interpretacyjnych',
    ' Thesis Conclusions': ' Zbiór finalnych dedukcji oraz implikacji',
    "This explanation shows model feature contributions, not real-world causes of this animal's outcome. Feature families like breed or coat color represent associations in the training set, not proof of direct impact.": 'Zobrazowana '
                                                                                                                                                                                                                             'mapa '
                                                                                                                                                                                                                             'sił '
                                                                                                                                                                                                                             'napędowych '
                                                                                                                                                                                                                             'dekomponuje '
                                                                                                                                                                                                                             'składowe '
                                                                                                                                                                                                                             'macierzy '
                                                                                                                                                                                                                             'i '
                                                                                                                                                                                                                             'ich '
                                                                                                                                                                                                                             'nacisk '
                                                                                                                                                                                                                             'punktowy '
                                                                                                                                                                                                                             'we '
                                                                                                                                                                                                                             'wnętrzu '
                                                                                                                                                                                                                             'architektury. '
                                                                                                                                                                                                                             'Nie '
                                                                                                                                                                                                                             'należy '
                                                                                                                                                                                                                             'ich '
                                                                                                                                                                                                                             'absolutnie '
                                                                                                                                                                                                                             'powiązywać '
                                                                                                                                                                                                                             'lub '
                                                                                                                                                                                                                             'dokonywać '
                                                                                                                                                                                                                             'uproszczeń '
                                                                                                                                                                                                                             'redukując '
                                                                                                                                                                                                                             'w '
                                                                                                                                                                                                                             'naturalne '
                                                                                                                                                                                                                             'związki '
                                                                                                                                                                                                                             'przyczynowo-skutkowe '
                                                                                                                                                                                                                             'charakteryzujące '
                                                                                                                                                                                                                             'otaczający '
                                                                                                                                                                                                                             'świat '
                                                                                                                                                                                                                             'empiryczny '
                                                                                                                                                                                                                             'i '
                                                                                                                                                                                                                             'reakcję '
                                                                                                                                                                                                                             'ludzi '
                                                                                                                                                                                                                             'w '
                                                                                                                                                                                                                             'fizycznym '
                                                                                                                                                                                                                             'ośrodku '
                                                                                                                                                                                                                             'adopcyjnym. '
                                                                                                                                                                                                                             'Oś '
                                                                                                                                                                                                                             'wektora '
                                                                                                                                                                                                                             'cech '
                                                                                                                                                                                                                             'taka '
                                                                                                                                                                                                                             'jak '
                                                                                                                                                                                                                             'przynależność '
                                                                                                                                                                                                                             'formalno-taksonomiczna '
                                                                                                                                                                                                                             'bądź '
                                                                                                                                                                                                                             'estetyka '
                                                                                                                                                                                                                             'umaszczenia '
                                                                                                                                                                                                                             'odzwierciedla '
                                                                                                                                                                                                                             'tu '
                                                                                                                                                                                                                             'twardo '
                                                                                                                                                                                                                             'korelacyjną '
                                                                                                                                                                                                                             'statystykę '
                                                                                                                                                                                                                             'macierzy '
                                                                                                                                                                                                                             'zbiorów '
                                                                                                                                                                                                                             'historycznych '
                                                                                                                                                                                                                             '(Dane '
                                                                                                                                                                                                                             'uczące '
                                                                                                                                                                                                                             'modelu) '
                                                                                                                                                                                                                             'w '
                                                                                                                                                                                                                             'izolacji '
                                                                                                                                                                                                                             'do '
                                                                                                                                                                                                                             'postulatów '
                                                                                                                                                                                                                             'by '
                                                                                                                                                                                                                             'stanowiły '
                                                                                                                                                                                                                             'one '
                                                                                                                                                                                                                             'obiektywne '
                                                                                                                                                                                                                             'oddziaływanie '
                                                                                                                                                                                                                             'bezpośrednie '
                                                                                                                                                                                                                             '(Impact).'}


def t(text: str) -> str:
    """Translate visible UI text for the selected language."""
    if st.session_state.get("language", "en") == "pl":
        return PL.get(text, text)
    return text


def localized_profile_label(label: str) -> str:
    """Translate common profile-label fragments while preserving model group names."""
    if st.session_state.get("language", "en") == "pl":
        replacements = {
            "Cat": "kot",
            "Dog": "pies",
            "has recorded name": "ma zapisane imię",
            "no recorded name": "brak zapisanego imienia",
            "normal": "normalny",
            "baby": "młode",
            "young": "młody dorosły",
            "adult": "dorosły",
            "senior": "senior",
            "unknown": "nieznany",
            "Stray": "Znalezione/bezdomne",
            "Owner Surrender": "Oddane przez właściciela",
            "Public Assist": "Pomoc publiczna",
            "Abandoned": "Porzucone",
            "Intact Male": "samiec niesterylizowany",
            "Intact Female": "samica niesterylizowana",
            "Neutered Male": "samiec kastrowany",
            "Spayed Female": "samica sterylizowana",
            "Unknown": "nieznane",
        }
    else:
        replacements = {
            "Cat": "cat",
            "Dog": "dog",
            "has recorded name": "recorded name present",
            "no recorded name": "no recorded name",
        }
    result = label
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result


st.set_page_config(
    page_title="AAC Adoption Thesis Demo",
    page_icon="",
    layout="wide",
)

# Inject premium CSS aesthetics
try:
    with open(PROJECT_ROOT / "assets" / "style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

@st.cache_data
def cached_tables() -> dict[str, pd.DataFrame]:
    return {
        "classification": load_table(TABLES_DIR, "classification"),
        "regression": load_table(TABLES_DIR, "regression"),
        "h1": load_table(TABLES_DIR, "h1"),
        "h3": load_table(TABLES_DIR, "h3"),
        "h5": load_table(TABLES_DIR, "h5"),
        "animal_archetypes": load_table(TABLES_DIR, "animal_archetypes"),
        "vulnerable_profiles": load_table(TABLES_DIR, "vulnerable_profiles"),
        "profile_contrasts": load_table(TABLES_DIR, "profile_contrasts"),
        "profile_model_error": load_table(TABLES_DIR, "profile_model_error"),
        "health_behavior_profiles": load_table(TABLES_DIR, "health_behavior_profiles"),
        "model_evidence_pack": load_table(TABLES_DIR, "model_evidence_pack"),
        "model_limitations_by_cohort": load_table(TABLES_DIR, "model_limitations_by_cohort"),
        "metric_confidence_intervals": load_table(TABLES_DIR, "metric_confidence_intervals"),
        "subgroup_reliability": load_table(TABLES_DIR, "subgroup_reliability"),
        "subgroup_metric_confidence_intervals": load_table(TABLES_DIR, "subgroup_metric_confidence_intervals"),
        "subgroup_adoption_milestones": load_table(TABLES_DIR, "subgroup_adoption_milestones"),
        "model_failure_modes": load_table(TABLES_DIR, "model_failure_modes"),
        "animal_journey_examples": load_table(TABLES_DIR, "animal_journey_examples"),
        "context_model_comparison": load_table(TABLES_DIR, "context_model_comparison"),
        "shap_classification": load_optional_csv(TABLES_DIR, "shap_global_classification.csv"),
        "shap_regression": load_optional_csv(TABLES_DIR, "shap_global_regression.csv"),
        "shap_family_classification": load_optional_csv(TABLES_DIR, "shap_feature_families_classification.csv"),
        "shap_family_regression": load_optional_csv(TABLES_DIR, "shap_feature_families_regression.csv"),
        "milestones": load_optional_csv(TABLES_DIR, "adoption_by_day_milestones.csv"),
    }


@st.cache_data
def cached_diagnostics() -> dict[str, pd.DataFrame]:
    return {
        "thresholds": load_diagnostic(DIAGNOSTICS_DIR, "thresholds"),
        "calibration": load_diagnostic(DIAGNOSTICS_DIR, "calibration"),
        "classification_slices": load_diagnostic(DIAGNOSTICS_DIR, "classification_slices"),
        "regression_slices": load_diagnostic(DIAGNOSTICS_DIR, "regression_slices"),
        "risk_quadrants": load_diagnostic(DIAGNOSTICS_DIR, "risk_quadrants"),
        "predictions": load_diagnostic(DIAGNOSTICS_DIR, "predictions"),
    }


def show_metric_cards(best_rows: pd.DataFrame) -> None:
    if best_rows.empty:
        st.info(t("Run the training and analysis pipeline to populate model comparison outputs."))
        return

    ordered = best_rows.sort_values(["task", "animal_subset"])
    columns = st.columns(min(len(ordered), 6))
    for column, (_, row) in zip(columns, ordered.iterrows()):
        metric_label = row["primary_metric"].upper()
        value = f"{row['score']:.3f}" if row["primary_metric"] != "mae" else f"{row['score']:.2f} days"
        column.metric(
            f"{row['animal_subset']} {row['task']}",
            value,
            help=f"{row['model_name']} by {metric_label}",
        )


def figure(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=t(caption), width='stretch')
    else:
        st.info(f"{t('Missing figure')}: {path.name}")


tables = cached_tables()
diagnostics = cached_diagnostics()
best_rows = best_model_rows(tables["classification"], tables["regression"])

selected_language = st.sidebar.selectbox(
    "Language / Język",
    list(LANGUAGES),
    index=0,
)
st.session_state["language"] = LANGUAGES[selected_language]

st.sidebar.divider()
if st.sidebar.button(t("Refresh Data"), key="refresh_data_btn"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    st.rerun()

st.title(t("AAC Adoption Thesis Demo"))
st.caption(t("Artifact-driven dashboard for model results, hypothesis signals, and model sensitivity checks."))

tabs = st.tabs(
    [
        t("Executive Overview"),
        t("Story Mode"),
        t("Animal Stories"),
        t("Model Quality"),
        t("Trust & Limits"),
        t("Interpretability"),
        t("Risk Explorer"),
        t("Hypothesis Lab"),
        t("Campaign Finder"),
        t("Model Sensitivity Demo"),
        t("Adoption Timeline"),
        t("Artifacts"),
        t("Context Data"),
        t(" Thesis Conclusions"),
    ]
)

with tabs[0]:
    st.markdown("##  Best Model Selection")
    st.write(t("The machine learning pipeline evaluated logistic regression, random forests, histogram gradient boosting, and CatBoost models. Here is the final selection based on empirical validation data:"))
    
    if best_rows.empty:
        st.info(t("Run the training and analysis pipeline to populate model comparison outputs."))
    else:
        clf_best = best_rows[best_rows["task"] == "classification"]
        reg_best = best_rows[best_rows["task"] == "regression"]
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("###  Classification (Adoption Chance)")
            if not clf_best.empty:
                row = clf_best.iloc[0]
                st.success(f"**Winner:** {row['model_name']}")
                st.metric(t("Primary Metric (PR-AUC)"), f"{row['score']:.3f}", help=t("Higher is better. Evaluated on out-of-time test set."))
                st.write(t("CatBoost consistently outperformed baseline models at separating adoptions from other outcomes, offering the highest precision-recall area under the curve (PR-AUC) for this imbalanced task."))
            else:
                st.warning(t("No classification artifacts found."))
                
        with col2:
            st.markdown("###  Regression (Wait Time)")
            if not reg_best.empty:
                row = reg_best.iloc[0]
                st.success(f"**Winner:** {row['model_name']}")
                st.metric(t("Primary Metric (MAE)"), f"{row['score']:.2f} days", help=t("Lower is better. Mean absolute error on test set."))
                st.write(t("For predicting the exact length of stay, CatBoost provided the lowest average error. However, length-of-stay is highly skewed and right-censored."))
            else:
                st.warning(t("No regression artifacts found."))
                
    st.divider()
    st.markdown("##  Where the Model is Wrong (Error Analysis)")
    st.write(t("No model is perfect. Here is exactly where the model struggles and the magnitude of its errors:"))
    
    err_col1, err_col2 = st.columns(2, gap="large")
    with err_col1:
        st.markdown("####  Classification Errors")
        st.write(t("When the model misclassifies an outcome, these are the most common failure modes:"))
        if not tables["model_failure_modes"].empty:
            st.dataframe(tables["model_failure_modes"].head(5), width='stretch', hide_index=True)
        else:
            st.info(t("Run `python scripts/generate_evidence_pack.py` to see failure modes."))
            
    with err_col2:
        st.markdown("####  Regression Magnitude Errors")
        st.write(t("The regression model's Mean Absolute Error (MAE) varies drastically by subgroup:"))
        if not diagnostics["regression_slices"].empty:
            _rs = diagnostics["regression_slices"]
            _cols = [c for c in ["cohort", "subgroup", "group", "mae", "mae_days", "records", "n"] if c in _rs.columns]
            st.dataframe(_rs[_cols].head(5), use_container_width=True, hide_index=True)
        else:
            st.info(t("Run `python scripts/generate_diagnostics.py` to see error slices."))
    
    st.divider()
    with st.expander(t("View Full Executive Summary Report")):
        st.markdown(load_summary(SUMMARY_DIR))

with tabs[1]:
    st.subheader(t("Data-to-Decision Story"))
    st.caption(t("How raw shelter records become thesis evidence and practical shelter-facing signals."))
    st.graphviz_chart(workflow_dot(), width='stretch')
    st.plotly_chart(decision_sankey(), width='stretch')

    st.subheader(t("Approach Comparison"))
    approaches = approach_comparison_rows()
    st.altair_chart(
        alt.Chart(approaches)
        .mark_bar()
        .encode(
            y=alt.Y("layer:N", sort=None, title=t("Analytical layer")),
            x=alt.X("count():Q", title=t("Story weight")),
            color=alt.Color("layer:N", legend=None),
            tooltip=["layer", "technology", "answers", "strength", "dashboard_use"],
        )
        .properties(height=280),
        width='stretch',
    )
    st.dataframe(approaches, width='stretch', hide_index=True)

    st.subheader(t("Real-life Shelter Questions"))
    card_columns = st.columns(5, gap="large")
    for column, card in zip(card_columns, story_cards()):
        column.metric(t(card["title"]), t(card["artifact"]))
        column.caption(t(card["question"]))

with tabs[2]:
    st.subheader(t("Animal Journey Cards"))
    archetypes = tables["animal_archetypes"]
    if archetypes.empty:
        st.info(t("Run `python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv` to populate animal stories."))
    else:
        labels = archetypes["profile_label"].head(250).tolist()
        selected_label = st.selectbox(t("Animal profile"), labels, format_func=localized_profile_label)
        selected = archetypes[archetypes["profile_label"].eq(selected_label)].iloc[0]
        profile_record = build_profile_prediction_record(selected)

        col1, col2, col3, col4 = st.columns(4, gap="large")
        col1.metric(t("Similar records"), f"{int(selected['records']):,}")
        col2.metric(t("Adoption rate"), f"{selected['adoption_rate_pct']:.1f}%")
        col3.metric(t("Median days to outcome"), f"{selected['median_days_to_outcome']:.1f} days")
        col4.metric(t("Visibility need"), selected["visibility_need"])

        tags = [
            t(str(selected.get('animal_type', ''))),
            t(str(selected.get('age_group', ''))),
            t(str(selected.get('intake_type', ''))),
            t(str(selected.get('intake_condition', ''))),
            t(str(selected.get('health_profile', 'unknown health'))),
            t(str(selected.get('behavior_support_flag', 'unknown behavior signal'))),
            str(selected.get('simplified_breed_group', '')),
            str(selected.get('simplified_color_group', '')),
            t('has recorded name') if selected.get('is_named') == True else t('no recorded name')
        ]
        
        # Filter out empty or 'None' values to keep tags clean
        clean_tags = [tag.strip() for tag in tags if tag and str(tag).strip().lower() not in ('none', 'nan', '')]

        tags_html = "".join([
            f"""<span style="display:inline-block; padding:0.4rem 0.8rem; margin:0.2rem; 
            border-radius:1rem; background-color:var(--secondary-background-color); 
            color:var(--text-color); font-size:0.85rem; font-weight:500; 
            border: 1px solid var(--faded-text-40); box-shadow: 0 1px 2px rgba(0,0,0,0.05);">{tag}</span>"""
            for tag in clean_tags
        ])
        
        st.markdown(f"""<div style="margin: 1rem 0 1.5rem 0;">
            <div style="font-weight: 600; margin-bottom: 0.5rem; font-size:1.05rem; color:var(--text-color);">{t('Profile')}:</div>
            <div style="display:flex; flex-wrap:wrap; margin-left:-0.2rem;">{tags_html}</div>
            </div>""", unsafe_allow_html=True)
        mix_cols = st.columns(3, gap="large")
        mix_cols[0].metric(t("Transfer rate"), f"{selected.get('transfer_rate_pct', 0):.1f}%")
        mix_cols[1].metric(t("Return-to-owner rate"), f"{selected.get('return_to_owner_rate_pct', 0):.1f}%")
        mix_cols[2].metric(t("Euthanasia rate"), f"{selected.get('euthanasia_rate_pct', 0):.1f}%")

        if st.button(t("Analyze Profile")):
            profile_prediction = None
            profile_similarity = similar_historical_cases(DATA_PATH, profile_record)
            try:
                profile_prediction = predict_from_record(profile_record, MODELS_DIR)
                if not profile_prediction.ok:
                    profile_prediction = None
            except Exception:
                profile_prediction = None

            st.subheader(t("Model View for This Journey"))
            if profile_prediction is None:
                st.info(t("Run `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv` to add representative CatBoost predictions to journey cards."))
            else:
                predicted_probability = profile_prediction.adoption_probability
                predicted_days = profile_prediction.predicted_days_to_outcome
                wait_bucket = profile_prediction.los_bucket
    
                model_cols = st.columns(4, gap="large")
                prob_label = t("Predicted adoption chance (calibrated)") if profile_prediction.is_calibrated else t("Predicted adoption chance")
                model_cols[0].metric(prob_label, f"{predicted_probability * 100:.1f}%")
                model_cols[1].metric(t("Predicted days to outcome"), f"{predicted_days:.1f} days")
                model_cols[2].metric(t("Length-of-stay bucket"), wait_bucket)
                model_cols[3].metric(t("Model visibility label"), t(visibility_need_from_prediction(predicted_probability, predicted_days)))
                with st.expander(t("Representative model record")):
                    st.dataframe(profile_record, width='stretch', hide_index=True)
    
            st.subheader(t("Similar Historical Cases"))
            if profile_similarity.empty:
                st.info(t("No similar historical cases found for this representative card."))
            else:
                st.dataframe(profile_similarity, width='stretch', hide_index=True)
    
            st.subheader(t("Top SHAP Reasons"))
            shap_view = pd.DataFrame()
            if profile_prediction is not None:
                try:
                    shap_view = local_shap_explanations(profile_record, MODELS_DIR, task="classification", top_n=8)
                except FileNotFoundError:
                    shap_view = pd.DataFrame()
            if shap_view.empty:
                shap_view = profile_global_shap_reasons(selected, tables["shap_classification"], top_n=8)
                if shap_view.empty:
                    st.info(t("Run `python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap` to populate SHAP reasons."))
                else:
                    st.caption(t("Model-wide SHAP signals mapped onto this animal profile; associations, not causes."))
                    st.dataframe(shap_view, width='stretch', hide_index=True)
            else:
                st.caption(t("Local CatBoost SHAP values for the representative journey record; associations, not causes."))
                st.dataframe(shap_view, width='stretch', hide_index=True)
    
            with st.expander(t(" Interpretation limits")):
                st.info(
                    t("This explanation shows model feature contributions, not real-world causes of this animal's outcome. "
                      "Feature families like breed or coat color represent associations in the training set, not proof of direct impact.")
                )

    st.subheader(t("Key Animal Contrasts"))
    contrasts = tables["profile_contrasts"]
    if not contrasts.empty:
        contrast_choice = st.selectbox(t("Contrast"), sorted(contrasts["contrast"].unique()))
        contrast_view = contrasts[contrasts["contrast"].eq(contrast_choice)]
        st.altair_chart(
            alt.Chart(contrast_view)
            .mark_bar()
            .encode(
                x=alt.X("contrast_value:N", title=t("Animal group")),
                y=alt.Y("adoption_rate_pct:Q", title=t("Adoption rate")),
                color="contrast_value:N",
                tooltip=["contrast_value", "records", "adoption_rate_pct", "median_days_to_outcome", "euthanasia_rate_pct"],
            )
            .properties(height=320),
            width='stretch',
        )
        st.dataframe(contrast_view, width='stretch', hide_index=True)
    left_animal, right_animal = st.columns(2, gap="large")
    with left_animal:
        figure(FIGURES_DIR / "animal_archetypes_top.png", "Largest animal archetypes")
    with right_animal:
        figure(FIGURES_DIR / "vulnerable_profiles.png", "Animal profiles needing visibility or support")
    st.subheader(t("Vulnerable Profiles"))
    st.dataframe(tables["vulnerable_profiles"].head(30), width='stretch', hide_index=True)
    st.subheader(t("Health and Behavior Support Profiles"))
    st.dataframe(tables["health_behavior_profiles"], width='stretch', hide_index=True)

with tabs[3]:
    left, right = st.columns(2, gap="large")
    with left:
        figure(FIGURES_DIR / "model_comparison_classification_pr_auc.png", "Classification PR-AUC")
        figure(FIGURES_DIR / "model_comparison_classification_roc_auc.png", "Classification ROC-AUC")
        figure(FIGURES_DIR / "model_comparison_classification_f1.png", "Classification F1")
    with right:
        figure(FIGURES_DIR / "model_comparison_regression_mae.png", "Regression MAE")
        figure(FIGURES_DIR / "model_comparison_regression_rmse.png", "Regression RMSE")

    st.subheader(t("Classification Table"))
    st.dataframe(tables["classification"], width='stretch', hide_index=True)
    st.subheader(t("Regression Table"))
    st.dataframe(tables["regression"], width='stretch', hide_index=True)

    st.subheader(t("Probability Trust Meter"))
    calibration = diagnostics["calibration"]
    if calibration.empty:
        st.info(t("Run `python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv` to populate calibration diagnostics."))
    else:
        st.altair_chart(
            alt.Chart(calibration)
            .mark_line(point=True)
            .encode(
                x=alt.X("mean_predicted_probability:Q", title=t("Mean predicted probability")),
                y=alt.Y("observed_adoption_rate:Q", title=t("Observed adoption rate")),
                tooltip=["probability_bin", "records", "mean_predicted_probability", "observed_adoption_rate"],
            )
            .properties(height=320),
            width='stretch',
        )
    st.subheader(t("Reliability Figures"))
    left_diag, right_diag = st.columns(2, gap="large")
    with left_diag:
        figure(FIGURES_DIR / "diagnostic_roc_curve.png", "Advanced model ROC curve")
        figure(FIGURES_DIR / "diagnostic_precision_recall_curve.png", "Advanced model precision-recall curve")
    with right_diag:
        figure(FIGURES_DIR / "diagnostic_calibration_curve.png", "Probability calibration")
        figure(FIGURES_DIR / "diagnostic_predicted_vs_actual.png", "Regression predicted vs actual")

with tabs[4]:
    render_trust_and_limits(tables)
    st.divider()
    st.subheader(t("Model Evidence Pack"))
    evidence = tables["model_evidence_pack"]
    intervals = tables["metric_confidence_intervals"]
    limitations = tables["model_limitations_by_cohort"]
    subgroup_reliability_table = tables["subgroup_reliability"]
    subgroup_intervals = tables["subgroup_metric_confidence_intervals"]
    subgroup_milestones = tables["subgroup_adoption_milestones"]
    failure_modes = tables["model_failure_modes"]
    journey_examples = tables["animal_journey_examples"]
    evidence_summary = load_summary(SUMMARY_DIR).split("## Model Evidence Pack", maxsplit=1)
    if evidence.empty and intervals.empty and limitations.empty:
        st.info(t("Run `python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv` to populate trust and limits artifacts."))
    else:
        if len(evidence_summary) > 1:
            st.markdown("## Model Evidence Pack" + evidence_summary[1])
        if not intervals.empty:
            st.subheader(t("Metric Confidence Intervals"))
            st.dataframe(intervals, width='stretch', hide_index=True)
            st.altair_chart(
                alt.Chart(intervals)
                .mark_rule(size=4)
                .encode(
                    y=alt.Y("metric:N", sort=None, title=t("Metric")),
                    x=alt.X("lower:Q", title=t("Bootstrap interval")),
                    x2="upper:Q",
                    color="animal_subset:N",
                    tooltip=["metric", "animal_subset", "lower", "estimate", "upper", "bootstrap_samples"],
                )
                .properties(height=260),
                width='stretch',
            )
        if not limitations.empty:
            st.subheader(t("Cohort Reliability Limits"))
            reliable = limitations[~limitations["small_cohort_flag"].fillna(False).astype(bool)] if "small_cohort_flag" in limitations.columns else limitations
            st.dataframe(reliable.head(30), width='stretch', hide_index=True)
            st.altair_chart(
                alt.Chart(reliable.head(30))
                .mark_bar()
                .encode(
                    x=alt.X("calibration_gap:Q", title=t("Calibration gap")),
                    y=alt.Y("value:N", sort="-x", title=t("Cohort value")),
                    color="cohort:N",
                    tooltip=["cohort", "value", "records", "calibration_gap", "mae", "false_negative_rate"],
                )
                .properties(height=360),
                width='stretch',
            )
        if not subgroup_reliability_table.empty:
            st.subheader(t("Subgroup Explorer"))
            subgroup_options = sorted(subgroup_reliability_table["cohort"].dropna().astype(str).unique().tolist())
            subgroup_choice = st.selectbox(t("Reliability subgroup"), subgroup_options)
            subgroup_view = subgroup_reliability_table[subgroup_reliability_table["cohort"].astype(str).eq(subgroup_choice)]
            stable_view = subgroup_view[~subgroup_view["small_cohort_flag"].fillna(False).astype(bool)] if "small_cohort_flag" in subgroup_view.columns else subgroup_view
            st.dataframe(stable_view, width='stretch', hide_index=True)
            st.altair_chart(
                alt.Chart(stable_view)
                .mark_bar()
                .encode(
                    x=alt.X("calibration_gap:Q", title=t("Calibration gap")),
                    y=alt.Y("value:N", sort="-x", title=subgroup_choice.replace("_", " ")),
                    tooltip=["value", "records", "observed_adoption_rate", "mean_predicted_adoption_probability", "calibration_gap", "mae"],
                )
                .properties(height=320),
                width='stretch',
            )
        if not failure_modes.empty:
            st.subheader(t("Where the Model Struggles"))
            st.dataframe(failure_modes.head(40), width='stretch', hide_index=True)
        if not subgroup_intervals.empty:
            st.subheader(t("Subgroup Metric Intervals"))
            interval_view = subgroup_intervals[subgroup_intervals["status"].eq("ok")] if "status" in subgroup_intervals.columns else subgroup_intervals
            st.dataframe(interval_view.head(50), width='stretch', hide_index=True)
        if not subgroup_milestones.empty:
            st.subheader(t("Time-to-Adoption Milestones"))
            milestone_group = st.selectbox(t("Milestone subgroup"), sorted(subgroup_milestones["cohort"].dropna().astype(str).unique().tolist()))
            milestone_view = subgroup_milestones[subgroup_milestones["cohort"].astype(str).eq(milestone_group)].head(20)
            milestone_chart = milestone_view.melt(
                id_vars=["value", "records", "adoptions", "adoption_rate_pct"],
                value_vars=["adopted_by_day_7_pct", "adopted_by_day_30_pct", "adopted_by_day_60_pct", "adopted_by_day_90_pct"],
                var_name="milestone",
                value_name="share",
            )
            st.altair_chart(
                alt.Chart(milestone_chart)
                .mark_bar()
                .encode(
                    x=alt.X("value:N", sort="-y", title=milestone_group.replace("_", " ")),
                    y=alt.Y("share:Q", title=t("Adopted animals (%)")),
                    color=alt.Color("milestone:N", title=t("Milestone")),
                    tooltip=[
                        alt.Tooltip("value:N", title=t("Value")),
                        alt.Tooltip("records:Q", title=t("Records")),
                        alt.Tooltip("adoptions:Q", title=t("Adoptions")),
                        alt.Tooltip("adoption_rate_pct:Q", title=t("Adoption rate")),
                        alt.Tooltip("milestone:N", title=t("Milestone")),
                        alt.Tooltip("share:Q", title=t("Adopted by day")),
                    ],
                )
                .properties(height=340),
                width='stretch',
            )
            st.dataframe(milestone_view, width='stretch', hide_index=True)
        if not journey_examples.empty:
            st.subheader(t("Animal Journey Evidence Examples"))
            st.dataframe(journey_examples, width='stretch', hide_index=True)

with tabs[5]:
    st.subheader(t("SHAP Global Explanations"))
    st.caption(t("SHAP values describe factors associated with model predictions, not causal effects."))
    st.info(SHAP_DISCLAIMER)
    left_shap, right_shap = st.columns(2, gap="large")
    with left_shap:
        figure(FIGURES_DIR / "shap_summary_classification.png", "Classification SHAP summary")
        st.dataframe(tables["shap_classification"].head(20), width='stretch', hide_index=True)
    with right_shap:
        figure(FIGURES_DIR / "shap_summary_regression.png", "Regression SHAP summary")
        st.dataframe(tables["shap_regression"].head(20), width='stretch', hide_index=True)
    st.subheader(t("Feature Family Scores"))
    family = tables["shap_family_classification"]
    if not family.empty:
        st.altair_chart(
            alt.Chart(family)
            .mark_bar()
            .encode(
                x=alt.X("mean_abs_shap:Q", title=t("Sum mean absolute SHAP")),
                y=alt.Y("feature_family:N", sort="-x", title=t("Feature family")),
                tooltip=["feature_family", "mean_abs_shap", "features"],
            )
            .properties(height=340),
            width='stretch',
        )
    else:
        st.info(t("Run diagnostics with `--include-shap` to populate interpretation artifacts."))

with tabs[6]:
    st.subheader(t("Risk Threshold Simulator"))
    thresholds = diagnostics["thresholds"]
    if thresholds.empty:
        st.info(t("Run diagnostics to populate threshold tradeoffs."))
    else:
        selected_threshold = st.slider(t("Adoption probability threshold"), 0.05, 0.95, 0.50, 0.05)
        selected = thresholds.iloc[(thresholds["threshold"] - selected_threshold).abs().argsort()[:1]]
        if not selected.empty:
            row = selected.iloc[0]
            cols = st.columns(4, gap="large")
            cols[0].metric(t("Precision"), f"{row['precision']:.3f}")
            cols[1].metric(t("Recall"), f"{row['recall']:.3f}")
            cols[2].metric(t("F1"), f"{row['f1']:.3f}")
            cols[3].metric(t("Flagged share"), f"{row['flagged_for_adoption_share']:.1%}")
        threshold_chart = thresholds.melt(
            id_vars=["threshold"],
            value_vars=["precision", "recall", "f1"],
            var_name="metric",
            value_name="value",
        )
        st.altair_chart(
            alt.Chart(threshold_chart)
            .mark_line(point=True)
            .encode(
                x=alt.X("threshold:Q", title=t("Threshold")),
                y=alt.Y("value:Q", title=t("Metric value")),
                color=alt.Color("metric:N", title=t("Metric")),
                tooltip=[
                    alt.Tooltip("threshold:Q", title=t("Threshold")),
                    alt.Tooltip("metric:N", title=t("Metric")),
                    alt.Tooltip("value:Q", title=t("Value")),
                ],
            )
            .properties(height=320),
            width='stretch',
        )

    st.subheader(t("Placement Risk Quadrant"))
    risk = diagnostics["risk_quadrants"]
    if not risk.empty:
        st.dataframe(risk, width='stretch', hide_index=True)
    predictions = diagnostics["predictions"]
    if not predictions.empty:
        st.altair_chart(
            alt.Chart(predictions.sample(min(len(predictions), 2000), random_state=42))
            .mark_circle(size=45, opacity=0.35)
            .encode(
                x=alt.X("predicted_adoption_probability:Q", title=t("Predicted adoption probability")),
                y=alt.Y("predicted_days_to_outcome:Q", title=t("Predicted days to outcome")),
                color="animal_type:N",
                tooltip=["animal_type", "age_group", "intake_type", "predicted_adoption_probability", "predicted_days_to_outcome"],
            )
            .properties(height=360),
            width='stretch',
        )
    st.subheader(t("Error Slice Explorer"))
    st.dataframe(diagnostics["classification_slices"].head(20), width='stretch', hide_index=True)
    st.dataframe(diagnostics["regression_slices"].head(20), width='stretch', hide_index=True)

with tabs[7]:
    h1_left, h1_right = st.columns(2, gap="large")
    with h1_left:
        figure(FIGURES_DIR / "h1_intake_type_adoption_rate.png", "H1 adoption rate by intake type")
        figure(FIGURES_DIR / "h3_age_group_adoption_rate.png", "H3 adoption rate by age group")
        figure(FIGURES_DIR / "h5_covid_period_adoption_rate.png", "H5 adoption rate by COVID period")
    with h1_right:
        figure(FIGURES_DIR / "h1_intake_condition_adoption_rate.png", "H1 adoption rate by intake condition")
        figure(FIGURES_DIR / "h3_age_group_median_days.png", "H3 median days by age group")
        figure(FIGURES_DIR / "h5_covid_period_median_days.png", "H5 median days by COVID period")

    st.subheader(t("H1: Intake vs Appearance"))
    st.dataframe(tables["h1"], width='stretch', hide_index=True)
    st.subheader(t("H3: Age and Length of Stay"))
    st.dataframe(tables["h3"], width='stretch', hide_index=True)
    st.subheader(t("H5: COVID-period Dynamics"))
    st.dataframe(tables["h5"], width='stretch', hide_index=True)

with tabs[8]:
    st.subheader(t("Campaign Candidate Finder"))
    st.caption(t("Exploratory cohort finder for groups that may benefit from targeted visibility. This is not causal recommendation logic."))
    predictions = diagnostics["predictions"]
    if predictions.empty:
        st.info(t("Run diagnostics to populate campaign cohorts."))
    else:
        filters = {}
        cols = st.columns(4, gap="large")
        for col, field in zip(cols, ["animal_type", "age_group", "intake_type", "covid_period"]):
            options = ["All"] + sorted(predictions[field].dropna().astype(str).unique().tolist())
            filters[field] = col.selectbox(t(field.replace("_", " ").title()), options, format_func=t)
        if st.button(t("Find Candidates")):
            cohort = predictions.copy()
            for field, value in filters.items():
                if value != "All":
                    cohort = cohort[cohort[field].astype(str).eq(value)]
            if cohort.empty:
                st.warning(t("No records match this cohort."))
            else:
                cols = st.columns(4, gap="large")
                cols[0].metric(t("Cohort size"), f"{len(cohort):,}")
                cols[1].metric(t("Observed adoption"), f"{cohort['classification_target'].mean() * 100:.1f}%")
                cols[2].metric(t("Mean predicted adoption"), f"{cohort['predicted_adoption_probability'].mean() * 100:.1f}%")
                cols[3].metric(t("Median predicted days"), f"{cohort['predicted_days_to_outcome'].median():.1f}")
                st.write(
                    t(
                        "Campaign framing: this cohort may be useful for targeted visibility when predicted adoption probability is low "
                        "or predicted days to outcome are high. Treat this as a prioritization signal, not proof of intervention impact."
                    )
                )
                st.dataframe(cohort.head(100), width='stretch', hide_index=True)

with tabs[9]:
    st.subheader(t("Model Sensitivity Demo"))
    st.warning(CAUSAL_WARNING)
    st.caption(t("Uses the combined classifier and regressor when advanced artifacts exist. This is a demo prediction, not a causal decision rule."))

    left, right = st.columns(2, gap="large")
    with left:
        animal_type = st.selectbox(t("Animal type"), ["Dog", "Cat"], format_func=t, key="pred_animal_type")
        intake_type = st.selectbox(
            t("Intake type"),
            ["Stray", "Owner Surrender", "Public Assist", "Abandoned", "Euthanasia Request"],
            format_func=t,
        )
        intake_condition = st.selectbox(
            t("Intake condition"),
            ["Normal", "Injured", "Sick", "Nursing", "Neonatal", "Aged", "Medical", "Behavior", "Other"],
            format_func=t,
        )
        sex_upon_intake = st.selectbox(
            t("Sex upon intake"),
            ["Intact Male", "Intact Female", "Neutered Male", "Spayed Female", "Unknown"],
            format_func=t,
        )
        has_name = st.toggle(t("Has name"), value=True)
    with right:
        age_years = st.slider(t("Age in years"), min_value=0.0, max_value=20.0, value=2.0, step=0.25)
        breed = st.text_input(t("Breed"), value="Labrador Retriever Mix" if animal_type == "Dog" else "Domestic Shorthair Mix")
        color = st.text_input(t("Color"), value="Black/White")
        intake_date = st.date_input(t("Intake date"), value=date(2024, 6, 1))

    record = build_prediction_record(
        animal_type=animal_type,
        intake_type=intake_type,
        intake_condition=intake_condition,
        sex_upon_intake=sex_upon_intake,
        age_days=age_years * 365.25,
        breed=breed,
        color=color,
        has_name=has_name,
        intake_date=pd.Timestamp(intake_date),
    )

    import hashlib
    record_hash = hashlib.md5(str(record.to_dict()).encode()).hexdigest()

    if st.button(t("Run prediction"), type="primary", key="run_prediction_btn"):
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("prediction_hash", None)
        try:
            prediction = predict_from_record(record, MODELS_DIR)
            st.session_state["prediction_result"] = prediction
            st.session_state["prediction_hash"] = record_hash
        except Exception as error:
            st.error(str(error))
            st.info(t("Run `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv` first."))

    if st.session_state.get("prediction_hash") == record_hash and "prediction_result" in st.session_state:
        prediction = st.session_state["prediction_result"]
        if not prediction.ok:
            st.error(prediction.error_message or t("Prediction failed."))
            st.info(t("Model output is unavailable. Check model artifacts and metadata."))
        else:
            probability_pct = prediction.adoption_probability * 100
            days = prediction.predicted_days_to_outcome
            wait_bucket = prediction.los_bucket

            col1, col2, col3 = st.columns(3, gap="large")
            prob_label = t("Predicted adoption probability (calibrated)") if prediction.is_calibrated else t("Predicted adoption probability")
            col1.metric(prob_label, f"{probability_pct:.1f}%")
            col2.metric(t("Predicted days to outcome"), f"{days:.1f} days")
            col3.metric(t("Length-of-stay bucket"), wait_bucket)
            st.dataframe(record, width='stretch', hide_index=True)
            similar = similar_historical_cases(DATA_PATH, record)
            if not similar.empty:
                st.subheader(t("Similar Historical Cases"))
                st.dataframe(similar, width='stretch', hide_index=True)
    elif "prediction_hash" in st.session_state and st.session_state["prediction_hash"] != record_hash:
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("prediction_hash", None)

with tabs[10]:
    st.subheader(t("Adoption Timeline"))
    milestones = tables["milestones"]
    if milestones.empty:
        st.info(t("Run diagnostics to generate adoption timeline milestones."))
    else:
        group = st.selectbox(t("Timeline group"), sorted(milestones["group"].unique()))
        view = milestones[milestones["group"].eq(group)].head(15)
        timeline_chart = view.melt(
            id_vars=["value", "adoptions"],
            value_vars=["adopted_by_day_7_pct", "adopted_by_day_30_pct", "adopted_by_day_90_pct"],
            var_name="milestone",
            value_name="share",
        )
        st.altair_chart(
            alt.Chart(timeline_chart)
            .mark_bar()
            .encode(
                x=alt.X("value:N", sort="-y", title=group.replace("_", " ")),
                y=alt.Y("share:Q", title=t("Share adopted (%)")),
                color=alt.Color("milestone:N", title=t("Milestone")),
                tooltip=[
                    alt.Tooltip("value:N", title=t("Value")),
                    alt.Tooltip("adoptions:Q", title=t("Adoptions")),
                    alt.Tooltip("milestone:N", title=t("Milestone")),
                    alt.Tooltip("share:Q", title=t("Adopted by day")),
                ],
            )
            .properties(height=360),
            width='stretch',
        )
        figure(FIGURES_DIR / "adoption_cumulative_curves.png", "Adoption timeline milestones")
        st.dataframe(milestones, width='stretch', hide_index=True)

with tabs[11]:
    st.subheader(t("Generated Artifacts"))

    # Manifest Explorer
    manifest_path = PROJECT_ROOT / "reports" / "artifact_manifest.csv"
    if manifest_path.exists():
        st.markdown(f"### {t('Artifact Manifest')}")
        st.write(t("All generated thesis deliverables, target definitions, and validation reports are listed below."))
        
        try:
            manifest_df = pd.read_csv(manifest_path)
            required_columns = {"artifact_path", "artifact_type", "required_for_thesis", "chapter", "notes", "source_script", "exists_on_disk"}
            missing_columns = sorted(required_columns - set(manifest_df.columns))
            if missing_columns:
                st.warning(f"Artifact manifest missing columns: {', '.join(missing_columns)}")
                manifest_df = pd.DataFrame(columns=sorted(required_columns))
            show_required_only = st.checkbox(t("Filter by Required for Thesis"), value=False)
            if show_required_only:
                manifest_df = manifest_df[manifest_df["required_for_thesis"].astype(str).isin(["True", "true", "1"])]
            
            # Map columns to localized versions
            display_df = manifest_df.copy()
            column_mapping = {
                "artifact_path": t("Artifact Path"),
                "artifact_type": t("Artifact Type"),
                "required_for_thesis": t("Required"),
                "chapter": t("Chapter"),
                "notes": t("Notes"),
                "source_script": t("Source Script"),
                "exists_on_disk": t("Exists"),
            }
            display_df = display_df.rename(columns=column_mapping)
            cols_to_show = [c for c in column_mapping.values() if c in display_df.columns]
            st.dataframe(display_df[cols_to_show], width='stretch', hide_index=True)
        except Exception as e:
            st.error(f"Error loading artifact manifest: {e}")
            
    # Report Reader
    st.markdown(f"### {t('Read Thesis & Methodology Reports')}")
    report_options = THESIS_REPORT_OPTIONS
    
    selected_report_path = st.selectbox(
        t("Select Report to View"),
        options=list(report_options.keys()),
        format_func=lambda x: t(report_options[x]),
    )
    
    if selected_report_path:
        report_file = PROJECT_ROOT / selected_report_path
        if report_file.exists():
            try:
                content = report_file.read_text(encoding="utf-8")
                st.divider()
                st.markdown(f"#### {t('Viewing Report:')} `{selected_report_path}`")
                st.markdown(content)
            except Exception as e:
                st.error(f"Error loading report: {e}")
        else:
            st.warning(t("Report file not found on disk."))

    st.divider()
    st.write(t("Core commands:"))
    st.code(
        "\n".join(
            [
                "python scripts/build_dataset.py --intakes data/raw/intakes.csv --outcomes data/raw/outcomes.csv --output data/processed/modeling_dataset.csv",
                "python scripts/run_eda.py --data data/processed/modeling_dataset.csv",
                "python scripts/train_baseline.py --data data/processed/modeling_dataset.csv",
                "python scripts/train_boosting.py --data data/processed/modeling_dataset.csv",
                "python scripts/train_advanced.py --data data/processed/modeling_dataset.csv",
                "python scripts/run_analysis.py --data data/processed/modeling_dataset.csv",
                "python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap",
                "python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv",
                "python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv",
                "python scripts/generate_report_outputs.py",
            ]
        ),
        language="bash",
    )
    st.write(t("Reports directory:"), str(PROJECT_ROOT / "reports"))
    st.write(t("Models directory:"), str(MODELS_DIR))

with tabs[12]:
    st.subheader(t("External Context Feature Test"))
    context_comparison = tables["context_model_comparison"]
    if context_comparison.empty:
        st.info(t("Run the commands below to populate `context_model_comparison.csv`."))
        st.code(
            "\n".join(
                [
                    "python scripts/download_context_data.py --output-dir data/raw/context",
                    "python scripts/build_dataset.py --intakes data/raw/intakes.csv --outcomes data/raw/outcomes.csv --output data/processed/modeling_dataset_context.csv --context-data-dir data/raw/context",
                    "python scripts/train_baseline.py --data data/processed/modeling_dataset.csv --metrics-dir reports/metrics_base --models-dir models/base_baseline --tables-dir reports/tables_base --output reports/metrics_base/baseline_metrics.csv",
                    "python scripts/train_boosting.py --data data/processed/modeling_dataset.csv --metrics-dir reports/metrics_base --models-dir models/base_boosting --tables-dir reports/tables_base",
                    "python scripts/train_advanced.py --data data/processed/modeling_dataset.csv --metrics-dir reports/metrics_base --models-dir models/base_advanced",
                    "python scripts/train_baseline.py --data data/processed/modeling_dataset_context.csv --metrics-dir reports/metrics_context --models-dir models/context_baseline --tables-dir reports/tables_context --output reports/metrics_context/baseline_metrics.csv",
                    "python scripts/train_boosting.py --data data/processed/modeling_dataset_context.csv --metrics-dir reports/metrics_context --models-dir models/context_boosting --tables-dir reports/tables_context",
                    "python scripts/train_advanced.py --data data/processed/modeling_dataset_context.csv --metrics-dir reports/metrics_context --models-dir models/context_advanced",
                    "python scripts/compare_context_models.py --base-metrics-dir reports/metrics_base --context-metrics-dir reports/metrics_context --tables-dir reports/tables",
                    "python scripts/generate_report_outputs.py",
                ]
            ),
            language="bash",
        )
    else:
        st.caption(t("Context features use intake-date weather and prior-window 311/intake-volume counts only."))
        view = context_comparison.copy()
        view["direction"] = view.apply(
            lambda row: "improved"
            if (row["delta"] > 0 and row["higher_is_better"] == True) or (row["delta"] < 0 and row["higher_is_better"] == False)
            else "worsened",
            axis=1,
        )
        view["direction_label"] = view["direction"].map(t)
        st.dataframe(view, width='stretch', hide_index=True)
        st.altair_chart(
            alt.Chart(view)
            .mark_bar()
            .encode(
                x=alt.X("delta:Q", title=t("Context minus base metric delta")),
                y=alt.Y("model_name:N", title=t("Model")),
                color=alt.Color("direction_label:N", title=t("Effect")),
                column=alt.Column("task:N", title=t("Task")),
                tooltip=["animal_subset", "model_name", "primary_metric", "base_score", "context_score", "delta", "direction_label"],
            )
            .properties(height=280),
            width='stretch',
        )

with tabs[13]:
    st.subheader(t(" Thesis Conclusions"))
    st.write(t("This dashboard translates raw shelter data into concrete evidence. Below are the finalized findings across the primary thesis hypotheses."))
    
    st.markdown("###  H1: Appearance vs. Intake Context")
    st.info(t("**Finding:** Physical appearance (Breed and Color) and Age are the strongest predictors of adoption, significantly outweighing the context of how an animal arrives (Intake Circumstances and Condition)."))
    h1_col1, h1_col2 = st.columns(2, gap="large")
    with h1_col1:
        if not tables["shap_family_classification"].empty:
            st.write(t("**Global SHAP Feature Importance (Classification)**"))
            view = tables["shap_family_classification"].head(5)
            st.altair_chart(
                alt.Chart(view).mark_bar().encode(
                    x=alt.X("mean_abs_shap:Q", title=t("Importance Impact")),
                    y=alt.Y("feature_family:N", sort="-x", title=""),
                    tooltip=["feature_family", "mean_abs_shap"]
                ).properties(height=200),
                width='stretch'
            )
    with h1_col2:
        if not tables["h1"].empty:
            st.write(t("**Adoption Rates by Intake Type**"))
            h1_view = tables["h1"][tables["h1"]["variable"] == "intake_type"].copy()
            h1_view = h1_view.rename(columns={"value": "intake_type"})
            st.dataframe(h1_view[["intake_type", "records", "adoption_rate_pct"]].sort_values("adoption_rate_pct", ascending=False).head(5), width='stretch', hide_index=True)

    st.markdown("###  H3: Age Penalties")
    st.warning(t("**Finding:** Older animals face significant penalties in adoption likelihood. Wait times to any outcome are complex, as seniors may leave the shelter faster due to higher rates of non-adoption outcomes."))
    h3_col1, h3_col2 = st.columns(2, gap="large")
    with h3_col1:
        if not tables["h3"].empty:
            st.write(t("**Median Days to Outcome by Age**"))
            st.altair_chart(
                alt.Chart(tables["h3"]).mark_bar(color="#ff7f0e").encode(
                    x=alt.X("age_group:N", sort=["baby", "young", "adult", "senior", "unknown"], title=""),
                    y=alt.Y("median_days_to_outcome:Q", title=t("Days")),
                    tooltip=["age_group", "median_days_to_outcome", "adoption_rate_pct"]
                ).properties(height=200),
                width='stretch'
            )
    with h3_col2:
        st.write(t("While puppies and kittens ('baby') often leave the shelter within 6-7 days, 'adult' animals face similar wait times. 'Senior' animals exhibit shorter median days to any outcome (e.g. 4.2 days), reflecting alternative outcome pathways. Age remains a critical predictive feature."))

    st.markdown("###  H5: COVID-19 Period Dynamics")
    st.success(t("**Finding:** The COVID-19 pandemic period was associated with a marked increase in adoption rates and a reduction in total volume. These period shifts must be accounted for to prevent model drift."))
    if not tables["h5"].empty:
        st.write(t("**Volume and Outcomes Across Periods**"))
        st.dataframe(tables["h5"][["covid_period", "records", "adoption_rate_pct", "median_days_to_outcome"]], width='stretch', hide_index=True)

    st.divider()
    st.markdown("###  Shelter Actionability & Limits")
    st.write(t("While machine learning successfully ranks animals by placement difficulty, **these predictions are associative, not causal.**"))
    cols = st.columns(3, gap="large")
    cols[0].metric(t("Best Use Case"), t("Prioritizing visibility campaigns"))
    cols[1].metric(t("Riskiest Use Case"), t("Automated euthanasia triaging"))
    cols[2].metric(t("Primary Limitation"), t("Data only reflects intake time"))

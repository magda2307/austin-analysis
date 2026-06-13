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

# Plain Polish interface copy. Keep technical terms only when they are needed to
# preserve the meaning of a metric, model output, or methodological limitation.
PL = {
    "AAC Adoption Thesis Demo": "Wyniki pracy magisterskiej: analiza adopcji w AAC",
    "Artifact-driven dashboard for model results, hypothesis signals, and model sensitivity checks.": "Panel przedstawia wyniki modeli, analizę hipotez i sprawdzenie wrażliwości modeli.",
    "Language": "Język",
    "Executive Overview": "Podsumowanie",
    "Story Mode": "Od danych do wniosków",
    "Animal Stories": "Przykłady zwierząt",
    "Model Quality": "Jakość modeli",
    "Trust & Limits": "Wiarygodność i ograniczenia",
    "Interpretability": "Interpretacja modeli",
    "Risk Explorer": "Analiza ryzyka",
    "Hypothesis Lab": "Hipotezy badawcze",
    "Campaign Finder": "Grupy do dodatkowej promocji",
    "Adoption Timeline": "Czas do adopcji",
    "Artifacts": "Pliki i raporty",
    "Context Data": "Dane kontekstowe",
    "Run the training and analysis pipeline to populate model comparison outputs.": "Uruchom proces trenowania i analizy, aby wygenerować porównanie modeli.",
    "Missing figure": "Brak wykresu",
    "Data-to-Decision Story": "Od danych do decyzji",
    "How raw shelter records become thesis evidence and practical shelter-facing signals.": "Jak dane schroniska są przetwarzane na wyniki badania i informacje przydatne w praktyce.",
    "Approach Comparison": "Porównanie podejść",
    "Analytical layer": "Rodzaj analizy",
    "Story weight": "Znaczenie w prezentacji",
    "Probability Trust": "Wiarygodność prawdopodobieństwa",
    "When model says 70% adoption chance, does reality agree?": "Czy w grupie z przewidywanym prawdopodobieństwem adopcji 70% rzeczywisty odsetek adopcji jest podobny?",
    "Calibration curve": "Krzywa kalibracji",
    "Long-stay Risk": "Ryzyko długiego pobytu",
    "Which animals look adoptable but may wait longer?": "Które zwierzęta mają wysokie przewidywane prawdopodobieństwo adopcji, ale mogą dłużej czekać na wynik?",
    "Model Failure Modes": "Najczęstsze błędy modeli",
    "Where do false negatives and large LOS errors cluster?": "W jakich grupach występuje najwięcej wyników fałszywie ujemnych i dużych błędów czasu pobytu?",
    "Which cohorts may deserve targeted visibility?": "Które grupy zwierząt mogą potrzebować dodatkowej promocji?",
    "Similar Cases": "Podobne przypadki",
    "What happened historically to animals like this one?": "Jakie wyniki odnotowano wcześniej dla zwierząt o podobnych cechach?",
    "Real-life Shelter Questions": "Pytania praktyczne",
    "Animal Journey Cards": "Przykładowe historie zwierząt",
    "Run `python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv` to populate animal stories.": "Uruchom `python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv`, aby wygenerować przykładowe historie zwierząt.",
    "Animal profile": "Profil zwierzęcia",
    "Similar records": "Podobne przypadki",
    "Adoption rate": "Odsetek adopcji",
    "Median days to outcome": "Mediana dni do dowolnego wyniku",
    "Visibility need": "Potrzeba dodatkowej promocji",
    "Profile": "Profil",
    "has recorded name": "ma zapisane imię",
    "no recorded name": "brak zapisanego imienia",
    "Transfer rate": "Odsetek przekazań",
    "Return-to-owner rate": "Odsetek zwrotów do właściciela",
    "Euthanasia rate": "Odsetek eutanazji",
    "Model View for This Journey": "Przewidywania modeli dla tego profilu",
    "Run `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv` to add representative CatBoost predictions to journey cards.": "Uruchom `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv`, aby dodać przewidywania CatBoost do przykładowych historii.",
    "Predicted adoption chance": "Przewidywane prawdopodobieństwo adopcji",
    "Predicted wait": "Przewidywany czas do dowolnego wyniku",
    "Predicted Time to Any Outcome": "Przewidywany czas do dowolnego wyniku",
    "Model visibility label": "Ocena potrzeby dodatkowej promocji",
    "Length-of-stay bucket": "Przedział przewidywanej długości pobytu",
    "Representative model record": "Dane przekazane do modelu",
    "high visibility need": "duża potrzeba dodatkowej promocji",
    "medium visibility need": "umiarkowana potrzeba dodatkowej promocji",
    "standard visibility": "standardowa promocja",
    "low": "niska",
    "medium": "umiarkowana",
    "high": "wysoka",
    "Similar Historical Cases": "Podobne przypadki historyczne",
    "No similar historical cases found for this representative card.": "Nie znaleziono podobnych przypadków historycznych dla tego profilu.",
    "Top SHAP Reasons": "Cechy o największym znaczeniu według SHAP",
    "Run `python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap` to populate SHAP reasons.": "Uruchom `python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap`, aby wygenerować wyniki SHAP.",
    "Model-wide SHAP signals mapped onto this animal profile; associations, not causes.": "Ogólne wyniki SHAP odniesione do tego profilu pokazują zależności w modelu, a nie przyczyny.",
    "Local CatBoost SHAP values for the representative journey record; associations, not causes.": "Lokalne wartości SHAP modelu CatBoost dla tego przypadku pokazują zależności w modelu, a nie przyczyny.",
    "Key Animal Contrasts": "Najważniejsze różnice między grupami",
    "Contrast": "Porównanie",
    "Animal group": "Grupa zwierząt",
    "Largest animal archetypes": "Najliczniejsze profile zwierząt",
    "Animal profiles needing visibility or support": "Profile zwierząt, które mogą potrzebować promocji lub wsparcia",
    "Vulnerable Profiles": "Profile wymagające szczególnej uwagi",
    "Health and Behavior Support Profiles": "Profile związane ze zdrowiem i zachowaniem",
    "Classification Table": "Wyniki klasyfikacji",
    "Regression Table": "Wyniki regresji",
    "Probability Trust Meter": "Ocena kalibracji prawdopodobieństwa",
    "Run `python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv` to populate calibration diagnostics.": "Uruchom `python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv`, aby wygenerować wyniki kalibracji.",
    "Mean predicted probability": "Średnie przewidywane prawdopodobieństwo",
    "Observed adoption rate": "Zaobserwowany odsetek adopcji",
    "Reliability Figures": "Wykresy jakości modeli",
    "Advanced model ROC curve": "Krzywa ROC modelu",
    "Advanced model precision-recall curve": "Krzywa precyzja-czułość modelu",
    "Probability calibration": "Kalibracja prawdopodobieństwa",
    "Regression predicted vs actual": "Wartości przewidywane i rzeczywiste w regresji",
    "Model Evidence Pack": "Zestaw wyników oceny modeli",
    "Run `python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv` to populate trust and limits artifacts.": "Uruchom `python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv`, aby wygenerować wyniki dotyczące wiarygodności i ograniczeń.",
    "Metric Confidence Intervals": "Przedziały ufności metryk",
    "Metric": "Metryka",
    "Bootstrap interval": "Przedział bootstrapowy",
    "Cohort Reliability Limits": "Ograniczenia jakości w poszczególnych grupach",
    "Calibration gap": "Różnica kalibracji",
    "Cohort value": "Wartość określająca grupę",
    "Subgroup Explorer": "Porównanie grup",
    "Reliability subgroup": "Analizowana grupa",
    "Where the Model Struggles": "Gdzie model popełnia większe błędy",
    "Subgroup Metric Intervals": "Przedziały metryk w grupach",
    "Time-to-Adoption Milestones": "Odsetek adopcji po określonej liczbie dni",
    "Milestone subgroup": "Analizowana grupa",
    "Adopted animals (%)": "Zwierzęta adoptowane (%)",
    "Milestone": "Punkt czasowy",
    "Value": "Wartość",
    "Records": "Liczba przypadków",
    "Adoptions": "Liczba adopcji",
    "Adopted by day": "Adoptowane do dnia",
    "Animal Journey Evidence Examples": "Przykłady historii zwierząt",
    "SHAP Global Explanations": "Ogólna interpretacja modelu za pomocą SHAP",
    "SHAP values describe factors associated with model predictions, not causal effects.": "Wartości SHAP opisują cechy związane z przewidywaniami modelu, a nie skutki przyczynowe.",
    "Classification SHAP summary": "Podsumowanie SHAP dla klasyfikacji",
    "Regression SHAP summary": "Podsumowanie SHAP dla regresji",
    "Feature Family Scores": "Znaczenie grup cech",
    "Sum mean absolute SHAP": "Suma średnich bezwzględnych wartości SHAP",
    "Feature family": "Grupa cech",
    "Run diagnostics with `--include-shap` to populate interpretation artifacts.": "Uruchom diagnostykę z opcją `--include-shap`, aby wygenerować wyniki interpretacji.",
    "Risk Threshold Simulator": "Wpływ zmiany progu klasyfikacji",
    "Run diagnostics to populate threshold tradeoffs.": "Uruchom diagnostykę, aby wygenerować wyniki dla różnych progów.",
    "Adoption probability threshold": "Próg prawdopodobieństwa adopcji",
    "Precision": "Precyzja",
    "Recall": "Czułość",
    "F1": "F1",
    "Flagged share": "Odsetek przypadków powyżej progu",
    "Threshold": "Próg",
    "Metric value": "Wartość metryki",
    "Placement Risk Quadrant": "Macierz prawdopodobieństwa adopcji i długości pobytu",
    "Predicted adoption probability": "Przewidywane prawdopodobieństwo adopcji",
    "Predicted days to outcome": "Przewidywana liczba dni do dowolnego wyniku",
    "Error Slice Explorer": "Błędy modelu w poszczególnych grupach",
    "H1: Intake vs Appearance": "H1: okoliczności przyjęcia i cechy zwierzęcia",
    "H3: Age and Length of Stay": "H3: wiek i długość pobytu",
    "H5: COVID-period Dynamics": "H5: zmiany w okresie pandemii COVID-19",
    "Campaign Candidate Finder": "Wyszukiwanie grup do dodatkowej promocji",
    "Exploratory cohort finder for groups that may benefit from targeted visibility. This is not causal recommendation logic.": "Narzędzie wyszukuje grupy, które mogą skorzystać z dodatkowej promocji. Wynik nie dowodzi, że promocja spowoduje adopcję.",
    "Run diagnostics to populate campaign cohorts.": "Uruchom diagnostykę, aby przygotować grupy do analizy promocji.",
    "All": "Wszystkie zwierzęta",
    "No records match this cohort.": "Żadne zwierzę nie spełnia wybranych warunków.",
    "Cohort size": "Liczba zwierząt w grupie",
    "Observed adoption": "Zaobserwowany odsetek adopcji",
    "Mean predicted adoption": "Średnie przewidywane prawdopodobieństwo adopcji",
    "Median predicted days": "Mediana przewidywanej liczby dni do dowolnego wyniku",
    "Campaign framing: this cohort may be useful for targeted visibility when predicted adoption probability is low or predicted days to outcome are high. Treat this as a prioritization signal, not proof of intervention impact.": "Ta grupa może wymagać dodatkowej promocji, gdy przewidywane prawdopodobieństwo adopcji jest niskie lub przewidywany czas do wyniku jest długi. Jest to sygnał pomocniczy, a nie dowód skuteczności promocji.",
    "Animal Type": "Gatunek",
    "Age Group": "Grupa wieku",
    "Intake Type": "Sposób przyjęcia",
    "Covid Period": "Okres względem pandemii COVID-19",
    "Uses the combined CatBoost classifier and regressor when advanced artifacts exist. This is a demo prediction, not a causal decision rule.": "Jeśli dostępne są modele CatBoost, narzędzie używa klasyfikatora i modelu regresyjnego dla wszystkich zwierząt. To demonstracja przewidywań, a nie przyczynowa reguła podejmowania decyzji.",
    "Animal type": "Gatunek",
    "Intake type": "Sposób przyjęcia",
    "Intake condition": "Stan przy przyjęciu",
    "Sex upon intake": "Płeć i status sterylizacji przy przyjęciu",
    "Has name": "Zapisane imię",
    "Age in years": "Wiek w latach",
    "Breed": "Rasa",
    "Color": "Umaszczenie",
    "Intake date": "Data przyjęcia",
    "Run prediction": "Oblicz przewidywanie",
    "Run `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv` first.": "Najpierw uruchom `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv`.",
    "Timeline group": "Grupa na osi czasu",
    "Run diagnostics to generate adoption timeline milestones.": "Uruchom diagnostykę, aby wygenerować wyniki czasu do adopcji.",
    "Share adopted (%)": "Odsetek adoptowanych (%)",
    "Adoption timeline milestones": "Odsetek adopcji w kolejnych dniach",
    "Generated Artifacts": "Wygenerowane pliki",
    "Core commands:": "Podstawowe polecenia:",
    "Reports directory:": "Katalog raportów:",
    "Models directory:": "Katalog modeli:",
    "External Context Feature Test": "Ocena dodatkowych danych kontekstowych",
    "Run the commands below to populate `context_model_comparison.csv`.": "Uruchom poniższe polecenia, aby utworzyć plik `context_model_comparison.csv`.",
    "Context features use intake-date weather and prior-window 311/intake-volume counts only.": "Dane kontekstowe obejmują pogodę z dnia przyjęcia oraz wcześniejsze liczby zgłoszeń 311 i przyjęć.",
    "Context minus base metric delta": "Różnica metryki po dodaniu danych kontekstowych",
    "Model": "Model",
    "Effect": "Zmiana",
    "Task": "Zadanie",
    "improved": "poprawa",
    "worsened": "pogorszenie",
    "Dog": "Pies",
    "Cat": "Kot",
    "Stray": "Zwierzę znalezione lub bezdomne",
    "Owner Surrender": "Oddane przez właściciela",
    "Public Assist": "Przyjęte w ramach pomocy publicznej",
    "Abandoned": "Porzucone",
    "Euthanasia Request": "Przyjęte na wniosek o eutanazję",
    "Normal": "Stan prawidłowy",
    "Injured": "Ranne",
    "Sick": "Chore",
    "Nursing": "Karmiące",
    "Neonatal": "Noworodek",
    "Aged": "W podeszłym wieku",
    "Medical": "Wymaga wsparcia medycznego",
    "Behavior": "Wymaga wsparcia behawioralnego",
    "Other": "Inne",
    "Intact Male": "Samiec niekastrowany",
    "Intact Female": "Samica niesterylizowana",
    "Neutered Male": "Samiec kastrowany",
    "Spayed Female": "Samica sterylizowana",
    "Unknown": "Nieznane",
    "unknown health": "brak informacji o zdrowiu",
    "unknown behavior signal": "brak informacji o zachowaniu",
    " Thesis Guide": "Przewodnik po pracy",
    "Model Sensitivity Demo": "Sprawdzenie wrażliwości modelu",
    "Filter by Required for Thesis": "Pokaż tylko pliki wymagane w pracy",
    "Select Report to View": "Wybierz raport",
    "Viewing Report:": "Wyświetlany raport:",
    "Report file not found on disk.": "Nie znaleziono pliku raportu.",
    "Target Definitions & Outcome Mappings": "Definicje zmiennych docelowych i wyników",
    "External Causal Validity & Generalisation Limits": "Możliwość uogólniania i ograniczenia wnioskowania przyczynowego",
    "Breed and Coat Colour Feature Engineering Justification": "Uzasadnienie cech rasy i umaszczenia",
    "Machine Learning vs. Descriptive Non-ML Baselines": "Modele uczenia maszynowego i opisowe punkty odniesienia",
    "H1 — Intake Profile & Causal Context": "H1 — Okoliczności przyjęcia i cechy zwierzęcia",
    "H2 — Seasonality & Intake Dynamics": "H2 — Sezonowość i liczba przyjęć",
    "H3 — Age and Time-to-Outcome Timing": "H3 — Wiek i czas do dowolnego wyniku",
    "H4 — Coat Colour (Black/Dark Animal Syndrome Check)": "H4 — Umaszczenie i hipoteza dotycząca ciemnego umaszczenia",
    "H5 — COVID-Period Population Shift and Volume Impact": "H5 — Zmiany liczby przyjęć i wyników w okresie pandemii",
    "H1 - Intake Profile & Causal Context": "H1 - Okoliczności przyjęcia i cechy zwierzęcia",
    "H2 - Seasonality & Intake Dynamics": "H2 - Sezonowość i liczba przyjęć",
    "H3 - Age and Time-to-Outcome Timing": "H3 - Wiek i czas do dowolnego wyniku",
    "H4 - Coat Colour (Black/Dark Animal Syndrome Check)": "H4 - Umaszczenie i hipoteza dotycząca ciemnego umaszczenia",
    "H5 - COVID-Period Population Shift and Volume Impact": "H5 - Zmiany w okresie pandemii COVID-19",
    "Hypothesis Evidence Matrix Summary": "Zestawienie wyników dla hipotez",
    "Data Leakage Audit & Control Log": "Kontrola wycieku informacji",
    "Propensity Score Matching Validation Examples": "Przykłady sprawdzenia dopasowania metodą propensity score",
    "Narrative Model Evidence & Key Findings Pack": "Podsumowanie wyników modeli",
    "Subgroup Reliability & Underrepresented Cohorts": "Jakość modeli w mniej licznych grupach",
    "Final Model Architecture Selection": "Wybór końcowych modeli",
    "Optimal Classification Threshold & Utility Analysis": "Wybór progu klasyfikacji",
    "Model Probability Calibration Interpretation": "Interpretacja kalibracji prawdopodobieństwa",
    "Operational Risk & Model Reliability Red Flags": "Ograniczenia i sygnały ryzyka modeli",
    "Data Pipeline Audit & Attrition Logging": "Kontrola przetwarzania danych i utraty rekordów",
    "Reproducibility Snapshot & Environment Info": "Informacje o środowisku i odtwarzalności",
    "quick placement likely": "prawdopodobny krótki pobyt",
    "needs visibility": "może potrzebować dodatkowej promocji",
    "long-stay risk": "ryzyko długiego pobytu",
    "outcome support priority": "priorytet wsparcia przed uzyskaniem wyniku",
    "Classification PR-AUC": "PR-AUC klasyfikacji",
    "Classification ROC-AUC": "ROC-AUC klasyfikacji",
    "Classification F1": "F1 klasyfikacji",
    "Regression MAE": "MAE regresji",
    "Regression RMSE": "RMSE regresji",
    "Predicted adoption chance (calibrated)": "Skalibrowane prawdopodobieństwo adopcji",
    "Predicted adoption probability (calibrated)": "Skalibrowane prawdopodobieństwo adopcji",
    "**Adoption Rates by Intake Type**": "**Odsetek adopcji według sposobu przyjęcia**",
    "**Finding:** Older animals face significant penalties in adoption likelihood. Wait times to any outcome are complex, as seniors may leave the shelter faster due to higher rates of non-adoption outcomes.": "**Wynik:** Starszy wiek wiązał się z niższym prawdopodobieństwem adopcji. Czas do dowolnego wyniku wymaga osobnej interpretacji, ponieważ starsze zwierzęta mogły szybciej opuszczać schronisko z powodu częstszych nieadopcyjnych wyników.",
    "**Finding:** Physical appearance (Breed and Color) and Age are the strongest predictors of adoption, significantly outweighing the context of how an animal arrives (Intake Circumstances and Condition).": "**Wynik:** Rasa, umaszczenie i wiek należały do najsilniejszych predyktorów adopcji i miały większe znaczenie predykcyjne niż sposób oraz stan przy przyjęciu.",
    "**Finding:** The COVID-19 pandemic period was associated with a marked increase in adoption rates and a reduction in total volume. These period shifts must be accounted for to prevent model drift.": "**Wynik:** Okres pandemii COVID-19 wiązał się z wyższym odsetkiem adopcji i mniejszą łączną liczbą przyjęć. Zmiany między okresami należy uwzględniać przy ocenie zmian jakości modelu w czasie.",
    "**Global SHAP Feature Importance (Classification)**": "**Ogólne znaczenie cech według SHAP dla klasyfikacji**",
    "**Median Days to Outcome by Age**": "**Mediana dni do dowolnego wyniku według wieku**",
    "**Volume and Outcomes Across Periods**": "**Liczba przyjęć i wyniki w kolejnych okresach**",
    "All generated thesis deliverables, target definitions, and validation reports are listed below.": "Poniżej wymieniono wygenerowane pliki pracy, definicje zmiennych docelowych i raporty walidacyjne.",
    "Artifact Manifest": "Wykaz plików",
    "Artifact Path": "Ścieżka pliku",
    "Artifact Type": "Typ pliku",
    "Automated euthanasia triaging": "Automatyczne podejmowanie decyzji o eutanazji",
    "Best Use Case": "Najlepsze zastosowanie",
    "CatBoost consistently outperformed baseline models at separating adoptions from other outcomes, offering the highest precision-recall area under the curve (PR-AUC) for this imbalanced task.": "CatBoost lepiej niż modele bazowe rozróżniał adopcje od pozostałych wyników i uzyskał najwyższą wartość PR-AUC w tym niezrównoważonym zadaniu.",
    "Chapter": "Rozdział",
    "Data only reflects intake time": "Dane obejmują wyłącznie informacje dostępne w chwili przyjęcia",
    "Days": "Dni",
    "Exists": "Istnieje",
    "For predicting the exact length of stay, CatBoost provided the lowest average error. However, length-of-stay is highly skewed and right-censored.": "W przewidywaniu długości pobytu CatBoost uzyskał najniższy średni błąd. Rozkład długości pobytu jest jednak silnie prawostronnie skośny, a część obserwacji jest prawostronnie cenzurowana.",
    "Higher is better. Evaluated on out-of-time test set.": "Wyższa wartość jest lepsza. Wynik obliczono na późniejszym chronologicznie zbiorze testowym.",
    "Importance Impact": "Znaczenie cechy",
    "Lower is better. Mean absolute error on test set.": "Niższa wartość jest lepsza. Jest to średni błąd bezwzględny na zbiorze testowym.",
    "Model output is unavailable. Check model artifacts and metadata.": "Wynik modelu jest niedostępny. Sprawdź pliki modelu i metadane.",
    "No classification artifacts found.": "Nie znaleziono wyników modelu klasyfikacyjnego.",
    "No model is perfect. Here is exactly where the model struggles and the magnitude of its errors:": "Poniżej pokazano grupy, w których model popełnia większe błędy, oraz skalę tych błędów.",
    "No regression artifacts found.": "Nie znaleziono wyników modelu regresyjnego.",
    "Notes": "Uwagi",
    "Prediction failed.": "Nie udało się obliczyć przewidywania.",
    "Primary Limitation": "Główne ograniczenie",
    "Primary Metric (MAE)": "Główna metryka: MAE",
    "Primary Metric (PR-AUC)": "Główna metryka: PR-AUC",
    "Prioritizing visibility campaigns": "Wybór zwierząt do dodatkowej promocji",
    "Read Thesis & Methodology Reports": "Raporty pracy i metodologii",
    "Refresh Data": "Odśwież dane",
    "Required": "Wymagany",
    "Riskiest Use Case": "Najbardziej ryzykowne zastosowanie",
    "Run `python scripts/generate_diagnostics.py` to see error slices.": "Uruchom `python scripts/generate_diagnostics.py`, aby zobaczyć błędy w poszczególnych grupach.",
    "Run `python scripts/generate_evidence_pack.py` to see failure modes.": "Uruchom `python scripts/generate_evidence_pack.py`, aby zobaczyć najczęstsze błędy modeli.",
    "Source Script": "Skrypt źródłowy",
    "The machine learning pipeline evaluated logistic regression, random forests, histogram gradient boosting, and CatBoost models. Here is the final selection based on empirical validation data:": "Porównano regresję logistyczną, lasy losowe, histogramowe wzmacnianie gradientowe i modele CatBoost. Poniżej przedstawiono wybór dokonany na podstawie wyników walidacyjnych.",
    "The regression model's Mean Absolute Error (MAE) varies drastically by subgroup:": "Średni błąd bezwzględny modelu regresyjnego różni się między grupami:",
    "This dashboard translates raw shelter data into concrete evidence. Below are the finalized findings across the primary thesis hypotheses.": "Panel przedstawia najważniejsze wyniki analiz dotyczących głównych hipotez pracy.",
    "Uses the combined classifier and regressor when advanced artifacts exist. This is a demo prediction, not a causal decision rule.": "Jeśli dostępne są odpowiednie pliki, narzędzie używa klasyfikatora i modelu regresyjnego dla wszystkich zwierząt. To demonstracja przewidywań, a nie przyczynowa reguła podejmowania decyzji.",
    "View Full Executive Summary Report": "Pokaż pełne podsumowanie",
    "When the model misclassifies an outcome, these are the most common failure modes:": "Poniżej przedstawiono najczęstsze rodzaje błędnej klasyfikacji:",
    "While machine learning successfully ranks animals by placement difficulty, **these predictions are associative, not causal.**": "Modele mogą porządkować zwierzęta według przewidywanej trudności uzyskania wyniku, ale **pokazują zależności predykcyjne, a nie przyczyny.**",
    "While puppies and kittens ('baby') often leave the shelter within 6-7 days, 'adult' animals face similar wait times. 'Senior' animals exhibit shorter median days to any outcome (e.g. 4.2 days), reflecting alternative outcome pathways. Age remains a critical predictive feature.": "Szczenięta i kocięta często opuszczały schronisko w ciągu 6–7 dni. Dorosłe zwierzęta miały podobny czas do wyniku. W grupie seniorów mediana czasu do dowolnego wyniku była krótsza, na przykład wynosiła 4,2 dnia, co może wynikać z częstszych wyników innych niż adopcja. Wiek pozostał ważną cechą predykcyjną.",
    " Interpretation limits": "Ograniczenia interpretacji",
    " Thesis Conclusions": "Wnioski z pracy",
    "This explanation shows model feature contributions, not real-world causes of this animal's outcome. Feature families like breed or coat color represent associations in the training set, not proof of direct impact.": "To wyjaśnienie pokazuje udział cech w przewidywaniu modelu, a nie rzeczywiste przyczyny wyniku danego zwierzęcia. Rasa i umaszczenie opisują zależności w danych treningowych, lecz nie dowodzą bezpośredniego wpływu.",
}


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

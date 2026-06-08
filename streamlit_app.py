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

PL = {
    "AAC Adoption Thesis Demo": "Demo pracy dyplomowej: adopcje AAC",
    "Artifact-driven dashboard for model results, hypothesis signals, and model sensitivity checks.": "Dashboard oparty na artefaktach: wyniki modeli, sygnaly hipotez i testy wrazliwosci modelu.",
    "Language": "Język",
    "Executive Overview": "Przegląd",
    "Story Mode": "Narracja",
    "Animal Stories": "Historie zwierząt",
    "Model Quality": "Jakość modelu",
    "Trust & Limits": "Zaufanie i ograniczenia",
    "Interpretability": "Interpretowalność",
    "Risk Explorer": "Eksplorator ryzyka",
    "Hypothesis Lab": "Laboratorium hipotez",
    "Campaign Finder": "Wyszukiwarka kampanii",
    "Adoption Timeline": "Oś czasu adopcji",
    "Artifacts": "Artefakty",
    "Context Data": "Dane kontekstowe",
    "Run the training and analysis pipeline to populate model comparison outputs.": "Uruchom pipeline trenowania i analizy, aby wypełnić wyniki porównania modeli.",
    "Missing figure": "Brak wykresu",
    "Data-to-Decision Story": "Od danych do decyzji",
    "How raw shelter records become thesis evidence and practical shelter-facing signals.": "Jak surowe rekordy schroniska stają się materiałem do pracy i praktycznymi sygnałami dla schroniska.",
    "Approach Comparison": "Porównanie podejść",
    "Analytical layer": "Warstwa analityczna",
    "Story weight": "Waga narracyjna",
    "Probability Trust": "Wiarygodność prawdopodobieństw",
    "When model says 70% adoption chance, does reality agree?": "Gdy model mówi o 70% szansy adopcji, czy rzeczywistość się zgadza?",
    "Calibration curve": "Krzywa kalibracji",
    "Long-stay Risk": "Ryzyko długiego pobytu",
    "Which animals look adoptable but may wait longer?": "Które zwierzęta wyglądają na adopcyjne, ale mogą czekać dłużej?",
    "Model Failure Modes": "Tryby błędów modelu",
    "Where do false negatives and large LOS errors cluster?": "Gdzie grupują się fałszywe negatywy i duże błędy długości pobytu?",
    "Which cohorts may deserve targeted visibility?": "Które kohorty mogą wymagać większej widoczności?",
    "Similar Cases": "Podobne przypadki",
    "What happened historically to animals like this one?": "Co historycznie działo się ze zwierzętami podobnymi do tego?",
    "Real-life Shelter Questions": "Praktyczne pytania schroniska",
    "Animal Journey Cards": "Karty ścieżki zwierzęcia",
    "Run `python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv` to populate animal stories.": "Uruchom `python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv`, aby wypełnić historie zwierząt.",
    "Animal profile": "Profil zwierzęcia",
    "Similar records": "Podobne rekordy",
    "Adoption rate": "Odsetek adopcji",
    "Median days to outcome": "Mediana dni do wyniku",
    "Visibility need": "Potrzeba widoczności",
    "Profile": "Profil",
    "has recorded name": "ma zapisane imię",
    "no recorded name": "brak zapisanego imienia",
    "Transfer rate": "Odsetek transferów",
    "Return-to-owner rate": "Odsetek powrotów do opiekuna",
    "Euthanasia rate": "Odsetek eutanazji",
    "Model View for This Journey": "Widok modelu dla tej ścieżki",
    "Run `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv` to add representative CatBoost predictions to journey cards.": "Uruchom `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv`, aby dodać predykcje CatBoost do kart ścieżki.",
    "Predicted adoption chance": "Prognozowana szansa adopcji",
    "Predicted wait": "Prognozowane oczekiwanie",
    "Predicted Time to Any Outcome": "Prognozowany czas do zakończenia pobytu",
    "Model visibility label": "Etykieta widoczności modelu",
    "Length-of-stay bucket": "Przedział długości pobytu",
    "Representative model record": "Reprezentatywny rekord modelu",
    "high visibility need": "wysoka potrzeba widoczności",
    "medium visibility need": "średnia potrzeba widoczności",
    "standard visibility": "standardowa widoczność",
    "low": "niska",
    "medium": "średnia",
    "high": "wysoka",
    "Similar Historical Cases": "Podobne przypadki historyczne",
    "No similar historical cases found for this representative card.": "Nie znaleziono podobnych przypadków historycznych dla tej karty.",
    "Top SHAP Reasons": "Najważniejsze powody SHAP",
    "Run `python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap` to populate SHAP reasons.": "Uruchom `python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap`, aby wypełnić powody SHAP.",
    "Model-wide SHAP signals mapped onto this animal profile; associations, not causes.": "Globalne sygnały SHAP przypisane do profilu; to powiązania, nie przyczyny.",
    "Local CatBoost SHAP values for the representative journey record; associations, not causes.": "Lokalne wartości SHAP CatBoost dla reprezentatywnego rekordu; to powiązania, nie przyczyny.",
    "Key Animal Contrasts": "Kluczowe kontrasty zwierząt",
    "Contrast": "Kontrast",
    "Animal group": "Grupa zwierząt",
    "Largest animal archetypes": "Największe archetypy zwierząt",
    "Animal profiles needing visibility or support": "Profile wymagające widoczności lub wsparcia",
    "Vulnerable Profiles": "Profile wrażliwe",
    "Health and Behavior Support Profiles": "Profile zdrowia i wsparcia behawioralnego",
    "Classification Table": "Tabela klasyfikacji",
    "Regression Table": "Tabela regresji",
    "Probability Trust Meter": "Miernik wiarygodności prawdopodobieństw",
    "Run `python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv` to populate calibration diagnostics.": "Uruchom `python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv`, aby wypełnić diagnostykę kalibracji.",
    "Mean predicted probability": "Średnie prognozowane prawdopodobieństwo",
    "Observed adoption rate": "Zaobserwowany odsetek adopcji",
    "Reliability Figures": "Wykresy wiarygodności",
    "Advanced model ROC curve": "Krzywa ROC modelu zaawansowanego",
    "Advanced model precision-recall curve": "Krzywa precision-recall modelu zaawansowanego",
    "Probability calibration": "Kalibracja prawdopodobieństw",
    "Regression predicted vs actual": "Regresja: prognoza vs wartość rzeczywista",
    "Model Evidence Pack": "Pakiet dowodów modelu",
    "Run `python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv` to populate trust and limits artifacts.": "Uruchom `python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv`, aby wypełnić artefakty zaufania i ograniczeń.",
    "Metric Confidence Intervals": "Przedziały ufności metryk",
    "Metric": "Metryka",
    "Bootstrap interval": "Przedział bootstrap",
    "Cohort Reliability Limits": "Ograniczenia wiarygodności kohort",
    "Calibration gap": "Luka kalibracji",
    "Cohort value": "Wartość kohorty",
    "Subgroup Explorer": "Eksplorator podgrup",
    "Reliability subgroup": "Podgrupa wiarygodności",
    "Where the Model Struggles": "Gdzie model ma trudności",
    "Subgroup Metric Intervals": "Przedziały metryk podgrup",
    "Time-to-Adoption Milestones": "Kamienie milowe czasu do adopcji",
    "Milestone subgroup": "Podgrupa kamieni milowych",
    "Adopted animals (%)": "Zwierzęta adoptowane (%)",
    "Milestone": "Kamień milowy",
    "Value": "Wartość",
    "Records": "Rekordy",
    "Adoptions": "Adopcje",
    "Adopted by day": "Adoptowane do dnia",
    "Animal Journey Evidence Examples": "Przykłady dowodów ścieżek zwierząt",
    "SHAP Global Explanations": "Globalne wyjaśnienia SHAP",
    "SHAP values describe factors associated with model predictions, not causal effects.": "Wartości SHAP opisują czynniki powiązane z predykcjami modelu, nie efekty przyczynowe.",
    "Classification SHAP summary": "Podsumowanie SHAP klasyfikacji",
    "Regression SHAP summary": "Podsumowanie SHAP regresji",
    "Feature Family Scores": "Wyniki rodzin cech",
    "Sum mean absolute SHAP": "Suma średnich bezwzględnych SHAP",
    "Feature family": "Rodzina cech",
    "Run diagnostics with `--include-shap` to populate interpretation artifacts.": "Uruchom diagnostykę z `--include-shap`, aby wypełnić artefakty interpretacji.",
    "Risk Threshold Simulator": "Symulator progu ryzyka",
    "Run diagnostics to populate threshold tradeoffs.": "Uruchom diagnostykę, aby wypełnić kompromisy progów.",
    "Adoption probability threshold": "Próg prawdopodobieństwa adopcji",
    "Precision": "Precyzja",
    "Recall": "Czułość",
    "F1": "F1",
    "Flagged share": "Odsetek oznaczonych",
    "Threshold": "Próg",
    "Metric value": "Wartość metryki",
    "Placement Risk Quadrant": "Kwadrant ryzyka umieszczenia",
    "Predicted adoption probability": "Prognozowane prawdopodobieństwo adopcji",
    "Predicted days to outcome": "Prognozowane dni do wyniku",
    "Error Slice Explorer": "Eksplorator przekrojów błędów",
    "H1: Intake vs Appearance": "H1: Przyjęcie vs wygląd",
    "H3: Age and Length of Stay": "H3: Wiek i długość pobytu",
    "H5: COVID-period Dynamics": "H5: Dynamika okresu COVID",
    "Campaign Candidate Finder": "Wyszukiwarka kandydatów do kampanii",
    "Exploratory cohort finder for groups that may benefit from targeted visibility. This is not causal recommendation logic.": "Eksploracyjna wyszukiwarka kohort, które mogą skorzystać z większej widoczności. To nie jest logika rekomendacji przyczynowej.",
    "Run diagnostics to populate campaign cohorts.": "Uruchom diagnostykę, aby wypełnić kohorty kampanii.",
    "All": "Wszystkie",
    "No records match this cohort.": "Brak rekordów pasujących do tej kohorty.",
    "Cohort size": "Wielkość kohorty",
    "Observed adoption": "Zaobserwowana adopcja",
    "Mean predicted adoption": "Średnia prognozowana adopcja",
    "Median predicted days": "Mediana prognozowanych dni",
    "Campaign framing: this cohort may be useful for targeted visibility when predicted adoption probability is low or predicted days to outcome are high. Treat this as a prioritization signal, not proof of intervention impact.": "Ujęcie kampanii: ta kohorta może być użyteczna dla działań zwiększających widoczność, gdy prognozowana szansa adopcji jest niska lub prognozowany czas do wyniku jest wysoki. Traktuj to jako sygnał priorytetyzacji, nie dowód wpływu interwencji.",
    "Animal Type": "Typ zwierzęcia",
    "Age Group": "Grupa wieku",
    "Intake Type": "Typ przyjęcia",
    "Covid Period": "Okres COVID",
    "Uses the combined CatBoost classifier and regressor when advanced artifacts exist. This is a demo prediction, not a causal decision rule.": "Używa połączonego klasyfikatora i regresora CatBoost, gdy istnieją artefakty modeli zaawansowanych. To predykcja demonstracyjna, nie przyczynowa reguła decyzyjna.",
    "Animal type": "Typ zwierzęcia",
    "Intake type": "Typ przyjęcia",
    "Intake condition": "Stan przy przyjęciu",
    "Sex upon intake": "Status płciowy przy przyjęciu",
    "Has name": "Ma imię",
    "Age in years": "Wiek w latach",
    "Breed": "Rasa",
    "Color": "Kolor",
    "Intake date": "Data przyjęcia",
    "Run prediction": "Uruchom predykcję",
    "Run `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv` first.": "Najpierw uruchom `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv`.",
    "Timeline group": "Grupa osi czasu",
    "Run diagnostics to generate adoption timeline milestones.": "Uruchom diagnostykę, aby wygenerować kamienie milowe adopcji.",
    "Share adopted (%)": "Odsetek adoptowanych (%)",
    "Adoption timeline milestones": "Kamienie milowe osi czasu adopcji",
    "Generated Artifacts": "Wygenerowane artefakty",
    "Core commands:": "Główne komendy:",
    "Reports directory:": "Katalog raportów:",
    "Models directory:": "Katalog modeli:",
    "External Context Feature Test": "Test cech kontekstowych",
    "Run the commands below to populate `context_model_comparison.csv`.": "Uruchom poniższe komendy, aby wypełnić `context_model_comparison.csv`.",
    "Context features use intake-date weather and prior-window 311/intake-volume counts only.": "Cechy kontekstowe używają pogody z dnia przyjęcia oraz wcześniejszych okien 311/liczby przyjęć.",
    "Context minus base metric delta": "Różnica metryki: kontekst minus baza",
    "Model": "Model",
    "Effect": "Efekt",
    "Task": "Zadanie",
    "improved": "poprawa",
    "worsened": "pogorszenie",
    "Dog": "Pies",
    "Cat": "Kot",
    "Stray": "Znalezione/bezdomne",
    "Owner Surrender": "Oddane przez właściciela",
    "Public Assist": "Pomoc publiczna",
    "Abandoned": "Porzucone",
    "Euthanasia Request": "Prośba o eutanazję",
    "Normal": "Normalny",
    "Injured": "Ranny",
    "Sick": "Chory",
    "Nursing": "Karmiący",
    "Neonatal": "Noworodek",
    "Aged": "Starszy",
    "Medical": "Medyczny",
    "Behavior": "Behawioralny",
    "Other": "Inny",
    "Intact Male": "Samiec niesterylizowany",
    "Intact Female": "Samica niesterylizowana",
    "Neutered Male": "samiec kastrowany",
    "Spayed Female": "samica sterylizowana",
    "Unknown": "nieznane",
    "unknown health": "nieznany stan zdrowia",
    "unknown behavior signal": "nieznany sygnał behawioralny",
    "📖 Thesis Guide": "Przewodnik po pracy",
    "Model Sensitivity Demo": "Demo wrażliwości modelu",
    "Generated Artifacts": "Wygenerowane artefakty",
    "Filter by Required for Thesis": "Filtruj: tylko wymagane do pracy",
    "Select Report to View": "Wybierz raport do wyświetlenia",
    "Viewing Report:": "Wyświetlany raport:",
    "Report file not found on disk.": "Plik raportu nie został znaleziony na dysku.",
    "Target Definitions & Outcome Mappings": "Definicje celów i mapowania wyników",
    "External Causal Validity & Generalisation Limits": "Zewnętrzna wiarygodność przyczynowa i granice generalizacji",
    "Breed and Coat Colour Feature Engineering Justification": "Uzasadnienie inżynierii cech rasy i maści",
    "Machine Learning vs. Descriptive Non-ML Baselines": "Uczenie maszynowe kontra opisowe linie bazowe bez ML",
    "H1 — Intake Profile & Causal Context": "H1 — Profil przyjęcia i kontekst przyczynowy",
    "H2 — Seasonality & Intake Dynamics": "H2 — Sezonowość i dynamika przyjęć",
    "H3 — Age and Time-to-Outcome Timing": "H3 — Wiek i czas do zakończenia pobytu",
    "H4 — Coat Colour (Black/Dark Animal Syndrome Check)": "H4 — Maść (Syndrom czarnego psa/kota)",
    "H5 — COVID-Period Population Shift and Volume Impact": "H5 — Zmiana populacji w okresie COVID i wpływ liczby przyjęć",
    "H1 - Intake Profile & Causal Context": "H1 - Profil przyjecia i kontekst przyczynowy",
    "H2 - Seasonality & Intake Dynamics": "H2 - Sezonowosc i dynamika przyjec",
    "H3 - Age and Time-to-Outcome Timing": "H3 - Wiek i czas do zakonczenia pobytu",
    "H4 - Coat Colour (Black/Dark Animal Syndrome Check)": "H4 - Masc (Syndrom czarnego psa/kota)",
    "H5 - COVID-Period Population Shift and Volume Impact": "H5 - Zmiana populacji w okresie COVID i wplyw liczby przyjec",
    "Hypothesis Evidence Matrix Summary": "Podsumowanie macierzy dowodów hipotez",
    "Data Leakage Audit & Control Log": "Audyt wycieku danych i dziennik kontroli",
    "Propensity Score Matching Validation Examples": "Przykłady walidacji dopasowania wskaźnika skłonności (PSM)",
    "Narrative Model Evidence & Key Findings Pack": "Pakiet opisowych dowodów modelu i kluczowych wniosków",
    "Subgroup Reliability & Underrepresented Cohorts": "Niezawodność w podgrupach i niedoreprezentowane kohorty",
    "Final Model Architecture Selection": "Wybór ostatecznej architektury modelu",
    "Optimal Classification Threshold & Utility Analysis": "Optymalny próg klasyfikacji i analiza użyteczności",
    "Model Probability Calibration Interpretation": "Interpretacja kalibracji prawdopodobieństwa modelu",
    "Operational Risk & Model Reliability Red Flags": "Ryzyko operacyjne i czerwone flagi niezawodności modelu",
    "Data Pipeline Audit & Attrition Logging": "Audyt potoku danych i logowanie atrycji",
    "Reproducibility Snapshot & Environment Info": "Migawka odtwarzalności i informacje o środowisku",
    "quick placement likely": "prawdopodobne szybkie umieszczenie",
    "needs visibility": "wymaga promowania",
    "long-stay risk": "ryzyko długiego pobytu",
    "outcome support priority": "priorytet wsparcia",
    "Classification PR-AUC": "Klasyfikacja PR-AUC",
    "Classification ROC-AUC": "Klasyfikacja ROC-AUC",
    "Classification F1": "Klasyfikacja F1",
    "Regression MAE": "Regresja MAE",
    "Regression RMSE": "Regresja RMSE",
    "Predicted adoption chance (calibrated)": "Prognozowana szansa adopcji (skalibrowana)",
    "Predicted adoption probability (calibrated)": "Prognozowane prawdopodobieństwo adopcji (skalibrowane)",
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
        t("🎓 Thesis Conclusions"),
    ]
)

with tabs[0]:
    st.markdown("## 🏆 Best Model Selection")
    st.write(t("The machine learning pipeline evaluated logistic regression, random forests, histogram gradient boosting, and CatBoost models. Here is the final selection based on empirical validation data:"))
    
    if best_rows.empty:
        st.info(t("Run the training and analysis pipeline to populate model comparison outputs."))
    else:
        clf_best = best_rows[best_rows["task"] == "classification"]
        reg_best = best_rows[best_rows["task"] == "regression"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎯 Classification (Adoption Chance)")
            if not clf_best.empty:
                row = clf_best.iloc[0]
                st.success(f"**Winner:** {row['model_name']}")
                st.metric(t("Primary Metric (PR-AUC)"), f"{row['score']:.3f}", help=t("Higher is better. Evaluated on out-of-time test set."))
                st.write(t("CatBoost consistently outperformed baseline models at separating adoptions from other outcomes, offering the highest precision-recall area under the curve (PR-AUC) for this imbalanced task."))
            else:
                st.warning(t("No classification artifacts found."))
                
        with col2:
            st.markdown("### ⏳ Regression (Wait Time)")
            if not reg_best.empty:
                row = reg_best.iloc[0]
                st.success(f"**Winner:** {row['model_name']}")
                st.metric(t("Primary Metric (MAE)"), f"{row['score']:.2f} days", help=t("Lower is better. Mean absolute error on test set."))
                st.write(t("For predicting the exact length of stay, CatBoost provided the lowest average error. However, length-of-stay is highly skewed and right-censored."))
            else:
                st.warning(t("No regression artifacts found."))
                
    st.markdown("---")
    st.markdown("## 📉 Where the Model is Wrong (Error Analysis)")
    st.write(t("No model is perfect. Here is exactly where the model struggles and the magnitude of its errors:"))
    
    err_col1, err_col2 = st.columns(2)
    with err_col1:
        st.markdown("#### ⚖️ Classification Errors")
        st.write(t("When the model misclassifies an outcome, these are the most common failure modes:"))
        if not tables["model_failure_modes"].empty:
            st.dataframe(tables["model_failure_modes"].head(5), width='stretch', hide_index=True)
        else:
            st.info(t("Run `python scripts/generate_evidence_pack.py` to see failure modes."))
            
    with err_col2:
        st.markdown("#### ⏱️ Regression Magnitude Errors")
        st.write(t("The regression model's Mean Absolute Error (MAE) varies drastically by subgroup:"))
        if not diagnostics["regression_slices"].empty:
            st.dataframe(diagnostics["regression_slices"][["cohort", "mae", "records"]].head(5), width='stretch', hide_index=True)
        else:
            st.info(t("Run `python scripts/generate_diagnostics.py` to see error slices."))
    
    st.markdown("---")
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
    card_columns = st.columns(5)
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
        profile_prediction = None
        profile_similarity = similar_historical_cases(DATA_PATH, profile_record)
        try:
            profile_prediction = predict_from_record(profile_record, MODELS_DIR)
            if not profile_prediction.ok:
                profile_prediction = None
        except Exception:
            profile_prediction = None

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(t("Similar records"), f"{int(selected['records']):,}")
        col2.metric(t("Adoption rate"), f"{selected['adoption_rate_pct']:.1f}%")
        col3.metric(t("Median days to outcome"), f"{selected['median_days_to_outcome']:.1f} days")
        col4.metric(t("Visibility need"), selected["visibility_need"])

        st.write(
            f"**{t('Profile')}:** {t(str(selected['animal_type']))} | {t(str(selected['age_group']))} | "
            f"{t(str(selected['intake_type']))} / {t(str(selected['intake_condition']))} | "
            f"{t(str(selected.get('health_profile', 'unknown health')))} | "
            f"{t(str(selected.get('behavior_support_flag', 'unknown behavior signal')))} | "
            f"{selected['simplified_breed_group']} / {selected['simplified_color_group']} | "
            f"{t('has recorded name') if selected['is_named'] == True else t('no recorded name')}"
        )
        mix_cols = st.columns(3)
        mix_cols[0].metric(t("Transfer rate"), f"{selected.get('transfer_rate_pct', 0):.1f}%")
        mix_cols[1].metric(t("Return-to-owner rate"), f"{selected.get('return_to_owner_rate_pct', 0):.1f}%")
        mix_cols[2].metric(t("Euthanasia rate"), f"{selected.get('euthanasia_rate_pct', 0):.1f}%")

        st.subheader(t("Model View for This Journey"))
        if profile_prediction is None:
            st.info(t("Run `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv` to add representative CatBoost predictions to journey cards."))
        else:
            predicted_probability = profile_prediction.adoption_probability
            predicted_days = profile_prediction.predicted_days_to_outcome
            wait_bucket = profile_prediction.los_bucket

            model_cols = st.columns(4)
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

        with st.expander(t("⚠️ Interpretation limits")):
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
    left_animal, right_animal = st.columns(2)
    with left_animal:
        figure(FIGURES_DIR / "animal_archetypes_top.png", "Largest animal archetypes")
    with right_animal:
        figure(FIGURES_DIR / "vulnerable_profiles.png", "Animal profiles needing visibility or support")
    st.subheader(t("Vulnerable Profiles"))
    st.dataframe(tables["vulnerable_profiles"].head(30), width='stretch', hide_index=True)
    st.subheader(t("Health and Behavior Support Profiles"))
    st.dataframe(tables["health_behavior_profiles"], width='stretch', hide_index=True)

with tabs[3]:
    left, right = st.columns(2)
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
    left_diag, right_diag = st.columns(2)
    with left_diag:
        figure(FIGURES_DIR / "diagnostic_roc_curve.png", "Advanced model ROC curve")
        figure(FIGURES_DIR / "diagnostic_precision_recall_curve.png", "Advanced model precision-recall curve")
    with right_diag:
        figure(FIGURES_DIR / "diagnostic_calibration_curve.png", "Probability calibration")
        figure(FIGURES_DIR / "diagnostic_predicted_vs_actual.png", "Regression predicted vs actual")

with tabs[4]:
    render_trust_and_limits(tables)
    st.markdown("---")
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
    left_shap, right_shap = st.columns(2)
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
            cols = st.columns(4)
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
    h1_left, h1_right = st.columns(2)
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
        cols = st.columns(4)
        for col, field in zip(cols, ["animal_type", "age_group", "intake_type", "covid_period"]):
            options = ["All"] + sorted(predictions[field].dropna().astype(str).unique().tolist())
            filters[field] = col.selectbox(t(field.replace("_", " ").title()), options, format_func=t)
        cohort = predictions.copy()
        for field, value in filters.items():
            if value != "All":
                cohort = cohort[cohort[field].astype(str).eq(value)]
        if cohort.empty:
            st.warning(t("No records match this cohort."))
        else:
            cols = st.columns(4)
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
    st.caption(t("Uses the combined CatBoost classifier and regressor when advanced artifacts exist. This is a demo prediction, not a causal decision rule."))

    left, right = st.columns(2)
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
        try:
            prediction = predict_from_record(record, MODELS_DIR)
            st.session_state["prediction_result"] = prediction
            st.session_state["prediction_hash"] = record_hash
        except Exception as error:
            st.error(str(error))
            st.info(t("Run `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv` first."))

    if st.session_state.get("prediction_hash") == record_hash and "prediction_result" in st.session_state:
        prediction = st.session_state["prediction_result"]
        probability_pct = prediction.adoption_probability * 100
        days = prediction.predicted_days_to_outcome
        wait_bucket = prediction.los_bucket
        
        col1, col2, col3 = st.columns(3)
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
                st.markdown("---")
                st.markdown(f"#### {t('Viewing Report:')} `{selected_report_path}`")
                st.markdown(content)
            except Exception as e:
                st.error(f"Error loading report: {e}")
        else:
            st.warning(t("Report file not found on disk."))

    st.markdown("---")
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
    st.subheader(t("🎓 Thesis Conclusions"))
    st.write(t("This dashboard translates raw shelter data into concrete evidence. Below are the finalized findings across the primary thesis hypotheses."))
    
    st.markdown("### 📌 H1: Appearance vs. Intake Context")
    st.info(t("**Finding:** The context of how an animal arrives (Intake Type and Condition) is a dramatically stronger predictor of adoption than physical appearance (Breed and Color)."))
    h1_col1, h1_col2 = st.columns(2)
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
            st.dataframe(tables["h1"][["intake_type", "records", "adoption_rate_pct"]].sort_values("adoption_rate_pct", ascending=False).head(5), width='stretch', hide_index=True)

    st.markdown("### 📌 H3: Age Penalties")
    st.warning(t("**Finding:** Older animals face severe penalties in both adoption likelihood and wait times, but the effect is non-linear and accelerates sharply for seniors."))
    h3_col1, h3_col2 = st.columns(2)
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
        st.write(t("While puppies and kittens ('baby') often leave the shelter within 5-7 days, 'adult' and 'senior' animals face wait times that are often 2x to 4x longer. Advanced models identify age as one of the top 3 critical predictive features."))

    st.markdown("### 📌 H5: COVID-19 Period Dynamics")
    st.success(t("**Finding:** The COVID-19 pandemic radically disrupted shelter operations, causing an artificial spike in adoption rates and a plunge in total volume that must be accounted for to prevent model drift."))
    if not tables["h5"].empty:
        st.write(t("**Volume and Outcomes Across Periods**"))
        st.dataframe(tables["h5"][["covid_period", "records", "adoption_rate_pct", "median_days_to_outcome"]], width='stretch', hide_index=True)

    st.markdown("---")
    st.markdown("### 🎯 Shelter Actionability & Limits")
    st.write(t("While machine learning successfully ranks animals by placement difficulty, **these predictions are associative, not causal.**"))
    cols = st.columns(3)
    cols[0].metric(t("Best Use Case"), t("Prioritizing visibility campaigns"))
    cols[1].metric(t("Riskiest Use Case"), t("Automated euthanasia triaging"))
    cols[2].metric(t("Primary Limitation"), t("Data only reflects intake time"))

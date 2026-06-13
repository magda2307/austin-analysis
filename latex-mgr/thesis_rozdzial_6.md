# 6. Architektura systemu i reprodukowalność eksperymentów

W niniejszym rozdziale zaprezentowano warstwę inżynieryjną i techniczną projektu. Współczesne badania oparte na uczeniu maszynowym wymagają nie tylko wykazania skuteczności predykcyjnej (opisanej w Rozdziale 5), ale również zapewnienia pełnej odtwarzalności (ang. *reproducibility*). Zgodnie z dobrymi praktykami oprogramowania naukowego (np. postulatami *The Turing Way*), rzetelne badanie powinno być możliwe do niezależnego odtworzenia na podstawie dostarczonego kodu i danych. W tym celu wdrożono rygorystyczną architekturę systemu, w pełni skonteneryzowane środowisko Docker, zautomatyzowane skrypty wykonawcze oraz kompleksowy zestaw testów.

## 6.1. Środowisko wdrożeniowe i konteneryzacja (Docker)

Fundamentem zapewnienia reprodukowalności wdrożonego rozwiązania jest jego skonteneryzowanie przy użyciu technologii **Docker**. Zamiast polegać na lokalnych, często różniących się środowiskach programistycznych (problem "u mnie działa"), cały system analityczny i aplikacyjny został spakowany w izolowany obraz.

### 6.1.1. Uzasadnienie złożoności systemu (Dlaczego MLOps?)

Mogłoby się wydawać, że do wytrenowania modelu uczenia maszynowego wystarczy pojedynczy, prosty skrypt w Pythonie. Skąd zatem potrzeba angażowania złożonej architektury Docker, izolowanych potoków i środowisk testowych? Wynika to bezpośrednio z natury problemu badawczego. 

Praca na 160 tysiącach epizodów zwierząt obejmujących 10 lat działalności schroniska niesie gigantyczne ryzyko tzw. **wycieku danych (data leakage)**. W amatorskim kodzie niezwykle łatwo jest o pomyłkę, w której np. podczas transformacji danych ujęto by przypadkiem informację z przyszłości (np. powód eutanazji jako predyktor, podczas gdy model ma działać na etapie wejścia zwierzęcia). Co więcej, zjawisko "cichej degradacji" (*silent failure*) oznacza, że w przypadku błędnie podanych danych model nie wygeneruje technicznego błędu (np. `Exception`), lecz bezgłośnie wyprodukuje fałszywą predykcję ryzyka. Zbudowanie rozbudowanej, rygorystycznej infrastruktury MLOps było absolutnie krytyczne, by udowodnić bezstronność i audytowalność wygenerowanych rezultatów, a nie tylko wysokie metryki same w sobie.

### 6.1.2. Architektura serwisów (Docker Compose)

Zastosowano narzędzie **Docker Compose** (`docker-compose.yml`), które definiuje architekturę systemu jako zbiór odseparowanych, opartych o wspólne zasoby kontenerów (serwisów). Taka struktura pozwala na natychmiastowe uruchomienie pożądanej warstwy systemu za pomocą jednej komendy. W ramach architektury zdefiniowano następujące serwisy operacyjne:

```yaml
services:
  dashboard:
    build: .
    ports: ["8501:8501"]
    volumes: [./models:/app/models]

  pipeline-full:
    build: .
    command: python scripts/run_full_pipeline.py
    volumes: [./data:/app/data, ./reports:/app/reports]

  test:
    build: .
    command: python -m pytest tests/ -v
```
*[Fragment Listing 6.1: Kluczowe definicje serwisów wyizolowane w architekturze docker-compose.yml]*

Wszystkie te serwisy współdzielą na poziomie kontenerów zmapowane wolumeny pamięci (katalogi `data/`, `models/`, `reports/`, `logs/`), co gwarantuje płynny przepływ wytworzonych artefaktów między poszczególnymi warstwami bez redundancji danych. Zwraca uwagę absolutne odseparowanie zapisu (`pipeline-full` montuje zapis do `/reports`) od konsumpcji (`dashboard` korzysta z `/models`).

## 6.2. Architektura "Artifact-First" i separacja warstw

Dzięki skonteneryzowaniu różnych ról systemu, możliwe stało się pełne zrealizowanie paradygmatu *Artifact-First*. Oznacza to ścisłe rozdzielenie zasobochłonnego procesu uczenia modeli od lekkiej warstwy prezentacyjnej (aplikacji dla użytkownika końcowego). 

Decyzja ta wynika z realnych ograniczeń środowiska docelowego. Typowe placówki ratownictwa zwierząt nie dysponują infrastrukturą serwerową wyposażoną w akceleratory graficzne (GPU) niezbędne do efektywnego trenowania modeli. Dodatkowo, interaktywne retrenowanie modelu wraz z wprowadzaniem każdego nowego zwierzęcia (tzw. uczenie online) niesie ze sobą ogromne ryzyko operacyjne — wprowadzony omyłkowo błędny rekord mógłby natychmiastowo zdegradować skuteczność całego systemu. 

* **Warstwa eksperymentalna (Jupyter):** Przeznaczona dla *data scientistów* do testowania hipotez.
* **Warstwa treningowa (Pipeline):** Z uwagi na duży wolumen danych (ponad 160 tys. epizodów) operacje te zajmują znaczny czas. Wyniki uruchomienia kontenera `pipeline-full` są "zamrażane" w formie statycznych artefaktów (wyuczone modele w formacie `.joblib`, tabele wyników w formacie `.csv` oraz wygenerowane wykresy `.png`).
* **Warstwa konsumpcyjna (Dashboard):** Aplikacja demonstracyjna (opisana w Rozdziale 7) nie posiada uprawnień do modyfikacji algorytmów. Jej rola sprowadza się wyłącznie do ładowania w pamięć RAM gotowych modeli z dysku. 

Takie podejście drastycznie obniża wymagania sprzętowe po stronie serwera aplikacji i gwarantuje niezmienność decyzji algorytmicznych – pracownik schroniska otrzymuje predykcję z modelu przetestowanego, a nie modelu zmutowanego najnowszymi danymi.

## 6.3. Organizacja kodu i modułowość

Projekt ustrukturyzowano jako standardowy pakiet języka Python zarządzany przez `pyproject.toml`. Struktura repozytorium odpowiada podziałowi kompetencji:

1. `src/aac_adoption/` – Główny kod źródłowy biblioteki, podzielony na domeny: `data` (przetwarzanie epizodów), `features` (rejestr cech), `models` (uczenie i ewaluacja) oraz `reporting` (agregacja wyników).
2. `scripts/` – Narzędzia wykonawcze (CLI), np. skrypt `run_full_pipeline.py`.
3. `tests/` – Zestaw zautomatyzowanych testów jednostkowych i integracyjnych uruchamianych z kontenera testowego.
4. `notebooks/` – Notatniki Jupyter służące analizie odkrywczej.

## 6.4. Automatyzacja potoku (Pipeline) a notatniki Jupyter

W uczeniu maszynowym częstym problemem jest zjawisko "ukrytego długu technicznego" (ang. *hidden technical debt*). Modele budowane wyłącznie poprzez sekwencyjne uruchamianie komórek w notatnikach Jupyter często stają się niemożliwe do odtworzenia. 

Dlatego w niniejszej architekturze Jupyter (`notebooks/`) służy wyłącznie do *badania i rozwoju* (R&D). Gdy logika okazuje się skuteczna, jest przenoszona do produkcyjnego kodu źródłowego w `src/`. Ostateczne trenowanie modeli realizowane jest przez w pełni skryptowy, centralny potok wywołujący wbudowany w kontener `pipeline-full`:
```bash
docker-compose up pipeline-full
```

### 6.4.1. Fazy zautomatyzowanego potoku i Data Provenance

Komenda ta zamyka proces badawczy w deterministycznych ramach, uruchamiając sekwencyjnie 18 zdefiniowanych kroków (widocznych w skrypcie `run_full_pipeline.py`). Proces ten można podzielić na następujące makro-fazy:
1. **Zarządzanie danymi:** Automatyczne pobranie surowych danych (jeśli ich brakuje), czyszczenie oraz budowa ujednoliconego zbioru (`build_dataset`).
2. **Modelowanie:** Trening modeli bazowych, strojenie hiperparametrów algorytmów drzewiastych, kalibracja klasyfikatorów.
3. **Diagnostyka i Wyjaśnialność:** Generowanie wykresów diagnostycznych i przeliczanie wartości SHAP dla całego zbioru.
4. **Raportowanie:** Automatyczne wypluwanie artefaktów (tabel i wykresów) bezpośrednio do katalogu `reports/`.

Szczególną uwagę w potoku poświęcono koncepcji **Data Provenance** (proweniencji danych). Po poprawnym zakończeniu wszystkich 18 kroków, potok generuje plik `run_receipt.json`. Pełni on funkcję "cyfrowego paragonu", w którym zapisywany jest dokładny znacznik czasu, skrót SHA z repozytorium Git określający wersję kodu, oraz pełna lista wykonanych kroków. Jest to ostateczny dowód na to, w jakich dokładnie warunkach powstały modele udostępniane później schronisku.

## 6.5. Rygor testowy i audyty danych

Złożoność przetwarzania danych weryfikowano na każdym etapie poprzez zautomatyzowane zestawy testów wchodzące w skład komendy `docker-compose up test`. System zwraca jednoznaczne logi udowadniające poprawność architektury:

```powershell
============================= test session starts ==============================
collected 24 items
tests/test_data_leakage.py::test_future_dates_not_in_features PASSED     [ 15%]
tests/test_data_leakage.py::test_target_variable_exclusion PASSED        [ 18%]
tests/test_model_pipeline.py::test_model_artifact_generation PASSED      [ 42%]
...
============================== 24 passed in 12.4s ==============================
```
*[Listing 6.2: Fragment wydruku środowiska testowego dokumentujący pomyślne zakończenie audytu wycieku danych (leakage audit)]*

Wszystkie niedeterministyczne operacje algorytmiczne w kodzie posiadają zamrożone ziarno losowości:
```python
# Utrzymanie globalnego, stałego ziarna dla procesów losowych
RANDOM_STATE = 42

# Przykład użycia w fazie podziału danych
train_df, val_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
```

W połączeniu z bezwzględną powtarzalnością architektury Docker, gwarantuje to, że uruchomienie potoku analitycznego na innej maszynie serwerowej wyprodukuje dokładnie takie same tabele wyników oraz wykresy, spełniając najsurowsze kryteria audytowalne prac naukowych.

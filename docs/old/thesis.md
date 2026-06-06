# Dane, które ratują życie: analiza i wizualizacja czynników wpływających na adopcję psów i kotów w Austin Animal Center z wykorzystaniem metod uczenia maszynowego

**Autorka:** Magdalena Sokołowska  
**Numer albumu:** S32924  
**Wydział:** Wydział Informatyki  
**Katedra:** Katedra Systemów Inteligentnych i Data Science  
**Kierunek:** Data Science  
**Promotor:** prof. dr hab. Grzegorz Wójcik  
**Miejsce i data:** Warszawa, styczeń 2026 r.  

## Streszczenie

Tutaj znajdzie się streszczenie całej pracy.

**Słowa kluczowe:** data science, koty, psy, adopcja, uczenie maszynowe, Austin Animal Center

## Abstract

**Life-Saving Data: Analyzing Factors Affecting Adoptions at the Austin Animal Center via Machine Learning and Visualization**

Keywords: data science, cats, dogs, adoption, machine learning, Austin Animal Center

## Spis treści

1. [Wstęp](#1-wstęp)
   1. [Kontekst problemu i znaczenie badań](#11-kontekst-problemu-i-znaczenie-badań)
   2. [Cel, hipotezy i pytania badawcze](#12-cel-hipotezy-i-pytania-badawcze)
   3. [Zakres i struktura pracy](#13-zakres-i-struktura-pracy)
2. [Podstawy teoretyczne i przegląd literatury](#2-podstawy-teoretyczne-i-przegląd-literatury)
   1. [Problem bezdomności zwierząt i filozofia No-Kill](#21-problem-bezdomności-zwierząt-i-filozofia-no-kill)
   2. [Predykcyjne modelowanie adopcji](#22-predykcyjne-modelowanie-adopcji)
   3. [Reprodukowalność i inżynieria danych](#23-reprodukowalność-i-inżynieria-danych)

## 1. Wstęp

### 1.1 Kontekst problemu i znaczenie badań

Każdego roku do schronisk dla zwierząt na całym świecie trafiają miliony bezdomnych psów i kotów, co stanowi ogromne wyzwanie organizacyjne i logistyczne. W samych Stanach Zjednoczonych szacuje się, że rocznie do schronisk trafia około 6 milionów psów i kotów [1]. Efektywne zarządzanie taką liczbą podopiecznych wymaga zapewnienia odpowiedniej opieki, zasobów i przestrzeni każdemu zwierzęciu, przy jednoczesnym dążeniu do jak najszybszego znalezienia mu nowego domu.

Jednym z kluczowych wskaźników skuteczności pracy schroniska jest czas pobytu zwierzęcia pod jego opieką, czyli Length of Stay (LOS). Im krótszy jest LOS, tym mniejsze obciążenie dla zwierzęcia, ponieważ ogranicza się stres, ryzyko chorób i ryzyko zgonu. Krótszy pobyt oznacza również mniejsze obciążenie dla placówki, niższe koszty utrzymania i więcej wolnego miejsca dla kolejnych potrzebujących zwierząt.

Coraz więcej schronisk przyjmuje filozofię No-Kill, która zakłada unikanie eutanazji zdrowych zwierząt poprzez osiąganie bardzo wysokiego odsetka adopcji i uratowań. Przykładem sukcesu podejścia No-Kill jest Austin Animal Center (AAC) w Teksasie. Od 2011 roku miasto Austin może poszczycić się mianem największego miasta w USA o statusie No-Kill, utrzymując współczynnik uratowań, czyli live-release rate, powyżej 90% [2]. Schronisko AAC przyjmuje ponad 18 tysięcy zwierząt rocznie [2], co czyni je jedną z największych tego typu placówek komunalnych w kraju i jednocześnie stawia przed nim ogromne wymagania w zakresie efektywnego wykorzystania danych i zasobów.

Dążenie do utrzymania wysokiego odsetka adopcji przy tak dużej skali działalności wymaga ciągłego usprawniania procesów i podejmowania trafnych decyzji operacyjnych. W ostatnich latach zauważalny jest istotny zwrot od podejmowania decyzji w oparciu o doświadczenie i intuicję personelu ku podejściu opartemu na danych. Większość dużych schronisk gromadzi szczegółowe informacje o każdym przyjęciu i opuszczeniu zwierzęcia, jednak w wielu przypadkach dane te nie są wykorzystywane do wyciągania praktycznych wniosków.

Zastosowanie nowoczesnych metod analizy danych i uczenia maszynowego otwiera możliwość lepszego zrozumienia, jakie czynniki sprzyjają szybkim adopcjom, a jakie wiążą się z dłuższym pobytem zwierząt w schronisku lub ze śmiercią. Modele predykcyjne mogą pomóc przewidywać prawdopodobieństwo adopcji konkretnego zwierzęcia oraz identyfikować obszary wymagające interwencji, na przykład grupy zwierząt o podwyższonym ryzyku długotrwałego pobytu.

Tym samym badania łączące dane schronisk z metodami uczenia maszynowego mają istotne znaczenie praktyczne. Mogą przyczynić się do optymalizacji pracy schronisk, lepszego alokowania zasobów oraz ratowania większej liczby zwierząt poprzez skracanie czasu oczekiwania na nowy dom i ograniczanie ryzyka śmierci.

### 1.2 Cel, hipotezy i pytania badawcze

Celem niniejszej pracy jest opracowanie systemu analitycznego wykorzystującego metody uczenia maszynowego do prognozowania adopcji zwierząt na podstawie danych historycznych ze schroniska oraz identyfikacji najważniejszych czynników wpływających na przebieg adopcji.

Praca skupia się na oszacowaniu prawdopodobieństwa adopcji oraz czasu oczekiwania na adopcję dla psów i kotów przyjętych do Austin Animal Center w latach 2013-2025. Osiągnięcie celu naukowego wymaga zbudowania dokładnego modelu predykcyjnego, ale równie ważne jest zadbanie o jego interpretowalność oraz użyteczność praktyczną dla nietechnicznych użytkowników.

Celem wdrożeniowym jest implementacja systemu, który integruje model w formie interaktywnej aplikacji demonstracyjnej. System ma umożliwiać eksplorację wyników predykcji, na przykład przewidywanego czasu adopcji danego zwierzęcia, oraz przeprowadzanie analiz typu „co-jeśli” w celu wsparcia decyzji dotyczących działań ukierunkowanych na zwierzęta o obniżonych szansach adopcyjnych.

Celem analitycznym jest pogłębiona identyfikacja cech zwierząt oraz czynników zewnętrznych, które istotnie wpływają na zmniejszenie prawdopodobieństwa adopcji. Wyniki mogą stanowić podstawę do dalszych interwencji, zmian w działaniach operacyjnych lub kampanii adopcyjnych ukierunkowanych na najbardziej narażone grupy.

#### Pytania badawcze

1. Które zmienne opisujące zwierzę oraz okoliczności jego przyjęcia do schroniska najsilniej wpływają na prawdopodobieństwo szybkiej adopcji?
2. Jak złożone modele uczenia maszynowego wypadają na tle prostszych metod w zadaniu przewidywania adopcji zwierząt?
3. Czy istnieją istotne różnice we wzorcach adopcji między psami a kotami oraz jak te wzorce zmieniały się w czasie?

#### Hipotezy badawcze

**H1:** Sposób, w jaki zwierzę trafia do schroniska, ma istotniejszy wpływ na prawdopodobieństwo i tempo jego adopcji niż cechy wyglądu. Zakłada się, że typ przyjęcia stanowi silniejszy prognostyk sukcesu adopcyjnego niż przynależność do konkretnej rasy.

**H2:** Występuje wyraźny efekt sezonowości w przebiegu adopcji. Pewne okresy w roku, na przykład miesiące letnie lub okolice przerw świątecznych, sprzyjają szybszym adopcjom, podczas gdy w innych okresach tempo adopcji spada.

**H3:** Wiek zwierzęcia ma istotny wpływ na jego szanse adopcyjne i długość pobytu w schronisku (`length of stay`). Młodsze zwierzęta są adoptowane szybciej niż starsze. Analiza prowadzona jest dwupoziomowo: (1) dla wszystkich zwierząt — predykcja `days_to_outcome` (czas do jakiegokolwiek wyniku), (2) tylko dla adoptowanych zwierząt — deskryptywna analiza `days_to_adoption` (czas do adopcji). Hipoteza H3 odnosi się do wzorców czasowych, nie do przyczynowości.

**H4:** Umaszczenie zwierzęcia wpływa na prawdopodobieństwo adopcji. Zwierzęta o ciemnym umaszczeniu, na przykład czarne psy i koty, są adoptowane rzadziej. W pracy zostanie zweryfikowane, czy w danych AAC obserwowalne jest zjawisko określane jako black dog syndrome.

**H5:** Pandemia COVID-19 miała istotny wpływ na dynamikę adopcji zwierząt, zarówno pod względem liczby adopcji, jak i długości pobytu. Zakłada się, że w okresie pandemicznym 2020-2021 nastąpiły zmiany we wzorcach adopcyjnych związane między innymi z pracą zdalną, ograniczeniami mobilności oraz kampaniami medialnymi promującymi adopcje.

Powyższe hipotezy zostaną poddane weryfikacji w dalszych rozdziałach pracy. Analiza ich trafności pozwoli nie tylko odpowiedzieć na postawione pytania badawcze, lecz także lepiej zrozumieć dynamikę procesów adopcyjnych.

### 1.3 Zakres i struktura pracy

---

**[SCOPE AND CLAIM — ENGLISH TECHNICAL REFERENCE FOR AI AGENTS AND REVIEWERS]**

**Official thesis claim (precise framing):**
This thesis builds a reproducible, leakage-safe, interpretable supervised machine learning pipeline for analyzing adoption likelihood and length-of-stay patterns in Austin Animal Center dog and cat records using intake-time features.

**What is and is not claimed:**
- The pipeline demonstrates **predictive association** between intake-time features and adoption outcomes / length of stay.
- It does NOT prove causation. No feature is claimed to *cause* adoption or non-adoption.
- The COVID-period variable (`covid_period`) captures a time-period label; differences across periods are associated with model output but do not imply COVID *caused* the changes.
- The regression target (`regression_target_days` = `days_to_outcome`) predicts **length of stay until any matched outcome**, not adoption speed. Adoption-only timing is analyzed separately as `days_to_adoption` (adopted animals only).
- Findings are descriptive evidence for the AAC dataset (2013–2025) only. They do not generalize beyond AAC without replication.
- The Streamlit dashboard is a thesis demo and presentation layer, not the main scientific contribution.

**Primary research questions (ML pipeline focus):**
1. Which intake-time features are most predictive of adoption likelihood (classification) and length of stay (regression)?
2. How do gradient boosting models compare to simpler baselines on AAC data?
3. What descriptive patterns are associated with H1 (intake type vs. appearance), H3 (age and time-to-outcome), and H5 (COVID-period dynamics)?

**Required terminology in all outputs:**
- Use: `predictive association`, `associated with`, `linked to model output`, `intake-time predictors`, `length of stay`, `time to outcome`, `descriptive time-to-adoption evidence`.
- Avoid: `causes adoption`, `proves animals are adopted faster because...`, `COVID caused...`, `black animals are discriminated against` (descriptive association framing only), `adoption speed` when the target is `days_to_outcome`.

**Target variable taxonomy:**
- `classification_target`: binary adoption indicator (1 = adoption outcome, 0 = other outcome)
- `regression_target_days` = `days_to_outcome`: length of stay until any matched outcome (main regression target)
- `days_to_adoption`: `days_to_outcome` restricted to adopted animals only (used only for H3 descriptive adopted-only analysis)
- Survival/time-to-event: not the main framework; descriptive KM curves provided; full survival modeling is future work

**Leakage control:** All model features are intake-time-only. Outcome fields (`outcome_type`, `outcome_datetime`, `sex_upon_outcome`, etc.) are used only as labels and targets, never as predictors. See `docs/target_definitions.md`.

**Dataset:** 162,390 matched intake/outcome episodes (dogs and cats, AAC 2013–2025). Each row is one shelter stay episode. Time-aware split: train 2013–2021, validation 2022–2023, test 2024–2025.

---

Zakres analizy obejmuje dane historyczne pochodzące z Austin Animal Center z lat 2013-2025. Dane zawierają informacje zarówno o przyjęciach zwierząt do schroniska, jak i o ich dalszych losach. Pozwalają prześledzić pełny cykl pobytu zwierzęcia w schronisku: od momentu znalezienia lub oddania, aż do chwili adopcji bądź innego zakończenia opieki.

W pracy skoncentrowano się na dwóch dominujących gatunkach, czyli psach i kotach, z pominięciem mniej licznych gatunków egzotycznych. Analizy zostaną przeprowadzone oddzielnie dla psów i kotów wszędzie tam, gdzie będzie to zasadne, aby uchwycić ewentualne różnice między tymi grupami.

Pod względem merytorycznym praca łączy zagadnienia z obszaru nauki o danych, uczenia maszynowego oraz inżynierii oprogramowania. Rozdział 2 przedstawia przegląd literatury oraz teoretyczne podstawy związane z problematyką bezdomności zwierząt i adopcji w kontekście zarządzania schroniskami. Rozdział 3 zawiera opis danych wykorzystanych w badaniu, źródła i charakterystykę zbioru danych z AAC, sposób ich wstępnego przetworzenia oraz analizę eksploracyjną.

Rozdział 4 przedstawia metodologię budowy modeli predykcyjnych, dobór cech, wybrane algorytmy uczenia maszynowego, strategie walidacji krzyżowej oraz metryki oceny. Szczególna uwaga zostanie poświęcona modelom typu gradient boosting, takim jak XGBoost i CatBoost, ze względu na ich skuteczność w zadaniach z danymi tabelarycznymi.

Rozdział 5 prezentuje wyniki eksperymentów, porównanie jakości predykcji różnych modeli oraz analizę interpretowalności najlepszego modelu. Wykorzystane zostaną techniki takie jak SHAP, które pozwalają określić wpływ poszczególnych cech na przewidywane prawdopodobieństwo adopcji.

Rozdział 6 opisuje podejście inżynierskie oraz elementy MLOps, w tym reprodukowalność eksperymentów, wersjonowanie danych i modeli oraz automatyzację procesu trenowania. Rozdział 7 przedstawia prototyp aplikacji demonstracyjnej w technologii Streamlit, służącej do prezentacji wyników modeli i analiz. Rozdział 8 zawiera podsumowanie, wnioski, weryfikację hipotez oraz propozycje dalszego rozwoju.

## 2. Podstawy teoretyczne i przegląd literatury

### 2.1 Problem bezdomności zwierząt i filozofia No-Kill

Zjawisko bezdomności zwierząt to poważny problem społeczny i ekologiczny na całym świecie. Szacuje się, że globalna populacja bezdomnych psów i kotów liczona jest w setkach milionów [4]. Szczególnie dotkliwie problem ten występuje w Stanach Zjednoczonych, gdzie każdego roku do schronisk trafia ponad 6 milionów porzuconych lub bezdomnych zwierząt domowych.

Historycznie wysoki odsetek zwierząt był poddawany eutanazji z braku miejsc i możliwości adopcji. W ostatnich latach, dzięki upowszechnieniu sterylizacji, kampaniom adopcyjnym oraz zmianie podejścia do zarządzania schroniskami, sytuacja zaczęła się stopniowo poprawiać. Trend ten odzwierciedla globalny zwrot w kierunku bardziej humanitarnego traktowania bezdomnych zwierząt, którego centralnym elementem jest filozofia No-Kill.

Filozofia No-Kill zakłada rezygnację z rutynowej eutanazji zwierząt w schronisku wyłącznie z powodu braku miejsc. Wszystkie zdrowe lub rokujące na wyleczenie zwierzęta powinny pozostać przy życiu i oczekiwać na adopcję tak długo, jak będzie to konieczne. W praktyce za schronisko no-kill uznaje się placówkę osiągającą co najmniej 90% współczynnika ocalenia, czyli save rate [2].

Przykładem skutecznej realizacji idei No-Kill jest miasto Austin w Teksasie. W 2010 roku władze Austin przyjęły kompleksowy plan działań, którego celem było osiągnięcie statusu no-kill, a już w kolejnym roku miejskie schronisko Austin Animal Center przekroczyło próg 90% uratowanych zwierząt [2]. Od tego czasu Austin utrzymuje reputację największego miasta No-Kill w USA.

Dane gromadzone w schroniskach stanowią cenne źródło informacji do oceny efektywności adopcji. Większość schronisk prowadzi szczegółową ewidencję przyjęć i opuszczeń zwierząt, co pozwala śledzić między innymi czas pobytu, stan zdrowia, rasę czy wiek. Analiza takich danych umożliwia wyliczenie kluczowych mierników operacyjnych i zastosowanie zaawansowanych metod analitycznych w celu predykcji wyników adopcji.

### 2.2 Predykcyjne modelowanie adopcji

Wyzwania związane z poprawą wskaźników adopcji przyczyniły się do rosnącego zainteresowania metodami predykcyjnymi. Modelowanie to polega na wykorzystaniu metod statystycznych oraz algorytmów uczenia maszynowego do przewidywania, czy i kiedy dane zwierzę zostanie adoptowane. Modele uczą się na danych historycznych w celu uchwycenia zależności między cechami zwierzęcia a wynikiem adopcyjnym [6].

Do najprostszych metod predykcji należą modele statystyczne, w tym regresja logistyczna. Umożliwia ona oszacowanie prawdopodobieństwa adopcji na podstawie zmiennych takich jak wiek, płeć czy rasa. Choć regresja cechuje się wysoką interpretowalnością, ma ograniczoną zdolność uchwycenia nieliniowych zależności.

Dlatego coraz częściej stosuje się metody o większej mocy predykcyjnej, takie jak zespoły drzew decyzyjnych, lasy losowe czy algorytmy z rodziny gradient boosting. Najbardziej rozpowszechnionym algorytmem tego typu jest XGBoost, wyróżniający się wydajnością w zadaniach klasyfikacyjnych [7]. W ostatnich latach popularność zyskał również CatBoost, biblioteka zoptymalizowana do przetwarzania cech kategorycznych [8].

W kontekście danych schronisk, które zawierają wiele atrybutów kategorycznych, takich jak rasa, umaszczenie czy typ przyjęcia, modele te są szczególnie przydatne. Badania z wykorzystaniem danych schroniskowych wykazały, że zaawansowane modele klasyfikacyjne potrafią z dużą dokładnością przewidywać szybkość adopcji, identyfikując wiek i rasę jako ważne predyktory sukcesu [6].

Istotnym elementem projektowania modeli jest także ocena istotności cech oraz interpretowalność. Złożone modele zespołowe często działają jak czarne skrzynki. Aby zrozumieć ich decyzje, stosuje się metody takie jak SHAP, czyli Shapley Additive Explanations [9]. Metoda ta pozwala określić, w jaki sposób konkretna cecha wpływa na obniżenie lub podwyższenie prawdopodobieństwa adopcji dla danego zwierzęcia.

### 2.3 Reprodukowalność i inżynieria danych

Wraz z rozwojem projektów z zakresu sztucznej inteligencji pojawiła się potrzeba wypracowania praktyk zapewniających reprodukowalność badań. Odpowiedzią na te wyzwania jest podejście MLOps, które integruje proces projektowania modeli z ich wdrażaniem i utrzymaniem [10].

Jednym z filarów MLOps jest zagwarantowanie spójnego środowiska uruchomieniowego. Służy do tego konteneryzacja, najczęściej realizowana z użyciem technologii Docker. Zapewnia ona, że kod trenujący model będzie działał identycznie niezależnie od środowiska, co jest kluczowe dla reprodukowalności badań naukowych [11].

Kolejnym ważnym elementem jest wersjonowanie danych. W przeciwieństwie do kodu, duże zbiory danych nie nadają się do przechowywania bezpośrednio w systemie Git. Dlatego w pracy wykorzystane zostanie narzędzie DVC, czyli Data Version Control, które umożliwia śledzenie zmian w zbiorach danych i modelach, zapewniając historię eksperymentów [12].

Ostatnim etapem będzie prezentacja wyników za pomocą biblioteki Streamlit, która pozwala na tworzenie interaktywnych wizualizacji danych oraz prototypowych aplikacji analitycznych [13].

## Cel kodu ML dla pracy

Kod tworzony w repozytorium powinien wspierać przede wszystkim cele badawcze pracy:

1. przygotowanie danych AAC o przyjęciach i wynikach adopcyjnych,
2. budowę zbioru modelowego dla psów i kotów,
3. predykcję prawdopodobieństwa adopcji,
4. predykcję czasu oczekiwania na adopcję,
5. porównanie modeli prostych z modelami gradient boosting,
6. analizę ważności cech i interpretowalność modelu,
7. weryfikację hipotez H1-H5,
8. przygotowanie wyników do wizualizacji i dashboardu.

### Priorytet hipotez w części analitycznej

Nie wszystkie hipotezy powinny mieć taki sam ciężar w implementacji i analizie, ponieważ mogłoby to rozproszyć strukturę pracy. Główna część analityczna powinna koncentrować się na trzech osiach:

1. **H1:** znaczenie typu przyjęcia w porównaniu z rasą, kolorem i innymi cechami wyglądu,
2. **H3:** wpływ wieku na tempo adopcji i długość pobytu,
3. **H5:** zmiana dynamiki adopcji w okresie COVID-19.

Hipotezy **H2** i **H4** powinny pełnić rolę uzupełniającą oraz opisową. Sezonowość i zjawisko black dog/cat syndrome warto uwzględnić w EDA i interpretacji, ale nie powinny dominować nad główną narracją modelowania.

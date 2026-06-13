# 4. Metodyka modelowania i oceny modeli

W rozdziale przedstawiono metodykę budowy i oceny modeli predykcyjnych. Problem sformułowano jako dwa odrębne zadania: klasyfikację binarną (adopcja względem innych wyników) oraz regresję (liczba dni do zakończenia pobytu). Odstąpiono od analizy przeżycia (ang. *survival analysis*) na rzecz prognozowania całkowitej długości pobytu (ang. *length of stay*). Opisano w nim podział zbioru, wybór algorytmów uczących oraz zastosowane metryki. Omówiono również metody interpretacji wyników za pomocą wartości SHAP oraz protokół badawczy zapewniający odtwarzalność obliczeń.

**Tabela 4.1.** Etapy procesu modelowania i oceny

| Etap badawczy | Działanie |
|---|---|
| 1. Przygotowanie cech | Wykorzystanie zbioru zdefiniowanego w rozdziale 3. Wykluczenie atrybutów rejestrowanych po przyjęciu w celu uniknięcia wycieku informacji (ang. *data leakage*). |
| 2. Podział danych | Podział chronologiczny: zbiór treningowy (2013–2021), walidacyjny (2022–2023) oraz testowy (2024–2025). |
| 3. Uczenie modeli | Trenowanie modeli bazowych i zespołowych osobno dla klasyfikacji i regresji. |
| 4. Ocena modeli | Ocena klasyfikacji za pomocą AUC-ROC, AUC-PR, F1-Score oraz regresji za pomocą MAE, RMSE i MedAE. |
| 5. Interpretacja | Analiza wyjaśnień globalnych i lokalnych za pomocą SHAP oraz wpływ zdefiniowanych Rodzin Cech. |
| 6. Reprodukowalność | Deterministyczna inicjalizacja algorytmów (`random_state=42`) oraz statyczny zapis wyników. |

## 4.1. Cel modelowania

Celem modelowania predykcyjnego jest prognozowanie wyników wyłącznie na podstawie informacji dostępnych w momencie przyjęcia zwierzęcia do schroniska. Analiza koncentruje się na mocy predykcyjnej, a nie na wnioskowaniu przyczynowo-skutkowym.

Celem modeli jest ustalenie, w jakim stopniu atrybuty takie jak gatunek, kategoria wiekowa czy okoliczności przyjęcia pozwalają przewidzieć dwie wartości: (1) prawdopodobieństwo adopcji zwierzęcia oraz (2) całkowity czas zajmowania miejsca w schronisku do momentu zakończenia pobytu (adopcja, transfer, powrót do właściciela). Podejście to dostarcza prognoz przydatnych w zarządzaniu placówką, w której czas pobytu każdego zwierzęcia generuje koszty operacyjne.

## 4.2. Formalne sformułowanie problemu

Modele uczono dla dwóch zadań:

1. **Klasyfikacja binarna (`classification_target`)**: Zmienna objaśniana \(Y^{(kl)} \in \{0,1\}\), gdzie \(Y^{(kl)}=1\) oznacza adopcję, a \(Y^{(kl)}=0\) inny wynik końcowy. Szacowane jest prawdopodobieństwo adopcji \(P(Y^{(kl)}=1\mid X)\).
2. **Regresja długości pobytu (`regression_target_days`)**: Zmienna ciągła wyrażona w dobach, reprezentująca czas do zdarzenia kończącego pobyt. Model estymuje wartość oczekiwaną długości trwania epizodu \(E(T\mid X)\).

Nie zastosowano modeli analizy przeżycia (takich jak model proporcjonalnych hazardów Coxa czy Random Survival Forest). Wymagałoby to skomplikowanej estymacji w obecności cenzurowania prawostronnego. Uczenie predykcyjne zdefiniowano jako klasyczną regresję na zamkniętych pobytach, na których można wprost zmierzyć długość pobytu. 

Zbiór predyktorów \(X\) wyklucza jakiekolwiek cechy będące składową wyniku końcowego.

## 4.3. Przygotowanie danych i podział chronologiczny

Dane mają charakter czasowy, dlatego zrezygnowano z klasycznej walidacji krzyżowej (ang. *k-fold cross-validation*) z tasowaniem próbek w celu uniknięcia wycieku informacji. Uczenie modelu na późniejszych rocznikach i testowanie go na wcześniejszych zawyżyłoby oceny skuteczności na historycznych wzorcach.

Zbiór 162 390 epizodów podzielono chronologicznie na trzy części:
- **Zbiór treningowy**: Lata 2013–2021. Pula obejmująca 126 759 rekordów psów i kotów.
- **Zbiór walidacyjny**: Lata 2022–2023. Obejmuje 21 811 epizodów. Przeznaczony do doboru parametrów modeli i wskaźników odcięcia.
- **Zbiór testowy**: Lata 2024–2025. Obejmuje 13 820 epizodów. Służy wyłącznie do ostatecznej oceny najlepszych algorytmów na najnowszych danych. 

Taki podział pozwala ocenić odporność modelu na zmiany wzorców w czasie (ang. *concept drift*), zwłaszcza po pandemii COVID-19.

## 4.4. Algorytmy predykcyjne

W badaniu porównano proste modele odniesienia z zaawansowanymi algorytmami zespołowymi. Pozwala to zmierzyć poprawę wyników wynikającą ze stosowania bardziej złożonych metod.

**Modele bazowe:**
- **Modele odniesienia (Dummy):** Przewidują większość (dla klasyfikacji) lub medianę (dla regresji). Wyznaczają minimalny poziom błędu w ujęciu deterministycznym.
- **Modele liniowe:** Regresja logistyczna dla binarnej zmiennej celu oraz regresja grzbietowa (Ridge) dla regresji długości pobytu. Oferują stabilność wyników oraz łatwość w analizie samych wag atrybutów.

**Modele zaawansowane (Drzewa decyzyjne):**
- **Lasy Losowe (Random Forest):** Model zespołowy zmniejszający wariancję przez uśrednienie prognoz.
- **Histogram-based Gradient Boosting:** Wariant sekwencyjnego uczenia błędów wykorzystujący kubełkowanie zmiennych numerycznych w celu skrócenia czasu uczenia dla dużych zbiorów.
- **CatBoost:** Architektura z rodziny wzmacniania gradientowego (ang. *Gradient Boosting*). Posiada mechanizmy przetwarzania zmiennych kategorycznych bez sztucznego wektoryzowania (*one-hot encoding*). Z tego powodu dobrze sprawdza się w przypadku setek nominalnych wartości ras i umaszczeń zwierząt.

Dla danych tabelarycznych o dużej liczbie zmiennych kategorycznych wybrano modele oparte na drzewach decyzyjnych (*gradient boosting*). Decyzja ta wynika z faktu, że algorytmy te stanowią branżowy standard (ang. *state-of-the-art*) dla tego typu danych. W przeciwieństwie do modeli liniowych, naturalnie wychwytują one nieliniowe interakcje między zmiennymi (np. specyficzny wpływ wieku tylko u konkretnej rasy psów) i nie wymagają rygorystycznego skalowania danych numerycznych. 

Zrezygnowano z zastosowania głębokich sieci neuronowych (ang. *Deep Learning*). Literatura badawcza i praktyka inżynieryjna dowodzą, że sieci neuronowe rzadko poprawiają jakość estymacji na niejednorodnych zbiorach tabelarycznych w porównaniu do algorytmów z rodziny *gradient boosting*, wymagając przy tym znacznie większych zasobów obliczeniowych i radykalnie obniżając łatwość interpretacji (tzw. problem czarnej skrzynki).

W procesie strojenia modeli (szczególnie CatBoost) przyjęto zachowawczą strategię doboru hiperparametrów, aby zapobiec zjawisku przeuczenia (ang. *overfitting*). Zastosowano umiarkowaną głębokość drzew decyzyjnych (`depth=6`), co ogranicza zapamiętywanie szumu z danych treningowych, oraz niski współczynnik uczenia (`learning_rate=0.05`), który pozwala na powolną, stabilną zbieżność algorytmu na przestrzeni 1000 iteracji. Dodatkowym mechanizmem obronnym było wdrożenie wczesnego zatrzymywania (`early_stopping_rounds=50`), które automatycznie przerywało proces uczenia, gdy metryka na zbiorze walidacyjnym przestała się poprawiać.
## 4.5. Metryki oceny

Zastosowano następujące metryki oceny:

**Zadanie klasyfikacji:**
- **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**: Miara ogólnej zdolności modelu do poprawnego rozróżniania klas. Wartość 1.0 oznacza idealne rozróżnianie, a 0.5 prognozę losową.
- **AUC-PR (Area Under the Precision-Recall Curve)**: Metryka użyteczna przy niewielkim braku równowagi klas, określająca sprawność detekcji kluczowej klasy.
- **F1-Score, Precyzja (Precision) i Czułość (Recall)**: Miary punktowe uzależnione od wyboru progu odcięcia (ang. *threshold*). W metodyce pracy zrezygnowano z naiwnego wyboru progu `0.5`. Optymalny próg decyzyjny dla każdego algorytmu został zoptymalizowany na niezależnym zbiorze walidacyjnym pod kątem maksymalizacji wskaźnika F1, a następnie przeniesiony na zbiór testowy w celu ewaluacji.
- **Ocena Kalibracji (Brier Score)**: Zdolność modelu do poprawnego rozróżniania klas (wysokie AUC) nie gwarantuje, że emitowane przez niego prawdopodobieństwa odzwierciedlają rzeczywiste prawdopodobieństwo sukcesu. W celu ewaluacji dokładności kalibracji wygenerowano wykresy kalibracyjne oraz zastosowano punktację Brier Score. Miara ta mierzy średni błąd średniokwadratowy prognozowanego prawdopodobieństwa w relacji do rzeczywistego wyniku. Jest to kluczowe z punktu widzenia budowy niezawodnego narzędzia analitycznego dla personelu schroniska.

**Zadanie regresji:**
- **MAE (Mean Absolute Error)**: Średni błąd bezwzględny, mierzony w dniach. Jest to główna metryka oceniająca średnią pomyłkę modelu.
- **MedAE (Median Absolute Error)**: Mediana z błędów bezwzględnych. Metryka bardzo odporna na błędy skrajne (np. zwierzęta pozostające w schronisku latami).
- **RMSE (Root Mean Square Error)**: Pierwiastek błędu średniokwadratowego. Stanowi karę za największe pomyłki modelu.

## 4.6. Interpretowalność (SHAP)

Wyniki modeli predykcyjnych w sektorze weterynaryjnym wymagają przejrzystości. Do analizy wpływu poszczególnych cech wykorzystano wartości SHAP (ang. *SHapley Additive exPlanations*).

Metoda ta wywodzi się z kooperatywnej teorii gier i pozwala oszacować wkład poszczególnych atrybutów w ostateczny wynik decyzyjny.
1. **Wyjaśnienia globalne (SHAP Summary):** Zestawienie pokazujące średni wpływ cech na decyzje modelu w całym zbiorze (np. korelacja młodego wieku z wyższym prawdopodobieństwem adopcji).
2. **Wyjaśnienia lokalne:** Ocena znaczenia atrybutów w konkretnym pojedynczym epizodzie. Funkcja ta dzieli prognozę na siły zwiększające lub zmniejszające końcowy wynik.
3. **Analiza Rodzin Cech:** Atrybuty poddano procesowi abstrakcji w "Rodziny Cech" (np. Wiek, Wygląd, Kontekst). Taki mechanizm umożliwia odrzucenie lub zachowanie pewnych grup zmiennych i sprawdzenie ogólnego spadku celności bez konieczności rozpatrywania wpływu indywidualnych wektorów.

## 4.7. Protokół eksperymentu

W celu zapewnienia odtwarzalności eksperymentu zastosowano następujące zasady:
1. **Jednolite ziarno losowości**: Ustalono deterministyczne ziarno liczb pseudolosowych (`random_state=42`) dla algorytmów.
2. **Rozdzielenie obliczeń i wizualizacji**: Proces uczenia odbywa się poprzez dedykowany skrypt (`scripts/run_full_pipeline.py`).
3. **Zapis modeli**: Wyniki modelowania i metadane ewaluacyjne zapisywane są bezpośrednio na dysk (jako estymatory `.joblib` oraz tabele `.csv`). Interfejs analityczny jedynie odczytuje parametry, bez konieczności kosztownego, ponownego uczenia modeli.

## 4.8. Ograniczenia

Zastosowane modele podlegają kilku ograniczeniom. Głównym technicznym obostrzeniem było wykluczenie nieukończonych pobytów na moment pobrania bazy (cenzorowanie prawostronne), konieczne do przeprowadzenia wariantu regresji długości pobytu. 

Ponadto modele odzwierciedlają statystyczne uwarunkowania lokalnego rynku w określonym przedziale czasu (szczególnie zasady AAC jako schroniska no-kill o wskaźniku przeżycia powyżej 90%). Prawidłowości przypisane np. do etykiet rasy mogą zależeć od lokalnej struktury społecznej i percepcji. Ich bezpośrednie przeniesienie do innych schronisk bez ponownego trenowania (kalibracji) mogłoby skutkować spadkiem skuteczności predykcyjnej.

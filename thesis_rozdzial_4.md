# 4. Metodyka modelowania i oceny modeli

W niniejszym rozdziale zaprezentowano ramy metodologiczne budowy oraz ewaluacji modeli predykcyjnych, których celem jest prognozowanie prawdopodobieństwa adopcji oraz estymacja całkowitego czasu pobytu zwierząt w schronisku. Problem badawczy sprowadzono do dwóch niezależnych zadań: klasyfikacji binarnej (adopcja względem innych wariantów zakończenia pobytu) oraz regresji (szacowanie liczby dni od przyjęcia do dowolnego wyniku końcowego). Zgodnie z założeniami opisanymi w poprzednim rozdziale, odstąpiono od klasycznego modelowania analizy przeżycia na rzecz bezpośredniej predykcji operacyjnej długości pobytu (ang. *Length of Stay*). Rozdział systematyzuje procedurę przygotowania danych, określa rygorystyczny podział chronologiczny zbioru, przedstawia dobór algorytmów uczących (w tym modeli liniowych i zespołowych, takich jak *Random Forest* i *CatBoost*) oraz definiuje zastosowane metryki oceny. Ponadto, omówiono metody interpretacji wyników w oparciu o wartości SHAP oraz opisano protokół badawczy gwarantujący pełną odtwarzalność zrealizowanych eksperymentów.

**Tabela 4.1.** Etapy procesu modelowania i ewaluacji

| Etap badawczy | Charakterystyka operacyjna |
|---|---|
| 1. Transformacja przestrzeni cech | Zastosowanie zbioru badawczego zdefiniowanego w rozdziale 3. Rygorystyczne wykluczenie atrybutów z przyszłości w celu uniknięcia wycieku informacji (ang. *data leakage*). |
| 2. Podział zbioru danych | Implementacja podziału chronologicznego: zbiór treningowy (2013–2021), walidacyjny (2022–2023) oraz testowy (2024–2025). |
| 3. Uczenie modeli | Estymacja parametryczna modeli bazowych oraz nieliniowych modeli zespołowych osobno dla zadania klasyfikacji i regresji w podgrupach gatunkowych. |
| 4. Walidacja wyników | Kwantyfikacja trafności za pomocą miar AUC-ROC, AUC-PR, F1-Score (dla klasyfikacji) oraz MAE, RMSE i MedAE (dla regresji). |
| 5. Interpretowalność procedur | Zastosowanie globalnych i lokalnych wyjaśnień SHAP oraz ewaluacja wpływu tzw. Rodzin Cech. |
| 6. Zapewnienie rygoru | Deterministyczna inicjalizacja algorytmów (`random_state=42`) oraz utrwalanie wyników na dysku, zapobiegające niekontrolowanemu retrenowaniu modeli w warstwie prezentacyjnej. |

## 4.1. Cel modelowania

Celem modelowania predykcyjnego w niniejszej pracy jest ewaluacja skuteczności prognozowania wyników na podstawie ograniczonych informacji dostępnych wyłącznie w momencie przyjęcia zwierzęcia do placówki. Zgodnie z literaturą przedmiotu w obszarze weterynaryjnego uczenia maszynowego, główny nacisk położono na zdolność klasyfikacyjną i regresyjną algorytmów na nowych danych, a nie na inferencję przyczynowo-skutkową (identyfikację bezpośrednich mechanizmów wpływu).

W badanej domenie problematyka sprowadza się do określenia, w jakim stopniu atrybuty takie jak gatunek, kategoria wiekowa, uwarunkowania przyjęcia czy okoliczności historyczne pozwalają przewidzieć dwie kwestie operacyjne: (1) prawdopodobieństwo pomyślnej adopcji zwierzęcia oraz (2) całkowity czas zajmowania boksu schroniskowego do momentu zakończenia epizodu, niezależnie od przyczyny tego finału (adopcja, transfer, zwrot). Zastosowanie klasycznego ujęcia predykcyjnego umożliwia pozyskanie rzetelnych prognoz na potrzeby zarządzania infrastrukturą ośrodka, gdzie każdy dzień pobytu generuje mierzalne koszty logistyczne.

## 4.2. Formalne sformułowanie problemu

Zgodnie z ustrukturyzowanymi w rozdziale 3. kontraktami danych, zdefiniowano dwa odizolowane zadania uczenia maszynowego:

1. **Klasyfikacja binarna (`classification_target`)**: Zmienna objaśniana \(Y^{(kl)} \in \{0,1\}\), gdzie \(Y^{(kl)}=1\) oznacza epizod zakończony adopcją, a \(Y^{(kl)}=0\) epizod zwieńczony dowolnym innym rezultatem. Estymowane jest prawdopodobieństwo a posteriori \(P(Y^{(kl)}=1\mid X)\).
2. **Regresja długości pobytu (`regression_target_days`)**: Zmienna ciągła wyrażona w dobach, obliczana jako różnica między czasem zamknięcia a czasem rozpoczęcia epizodu. Reprezentuje czas do zaistnienia dowolnego zdarzenia kończącego pobyt. Model estymuje wartość oczekiwaną długości trwania epizodu \(E(T\mid X)\).

Zdecydowano o odstąpieniu od stosowania formalnych modeli analizy przetrwania (takich jak model proporcjonalnych hazardów Coxa czy Random Survival Forest) jako rdzenia predykcyjnego. Podejście to nastręcza trudności związanych z estymacją rozkładów w obecności prawostronnego cenzurowania w dynamicznym środowisku. W toku budowy systemu ML zrezygnowano z uwzględniania otwartych (cenzurowanych) pobytów, traktując przewidywanie jako standardowy wariant regresji.

Zbiór predyktorów \(X\) dla każdego z wariantów zdefiniowano z rygorystycznym uwzględnieniem zasady przyczynowości czasowej, wykluczając jakiekolwiek cechy będące następstwem lub składową definiowanego wyniku końcowego.

## 4.3. Przygotowanie danych i podział chronologiczny

Z uwagi na temporalny charakter danych schroniskowych, rezygnacja z klasycznej walidacji krzyżowej (*k-fold cross-validation*) ze zjawiskiem losowego tasowania (*shuffling*) stanowiła konieczność chroniącą przed przeciekiem danych. Trenowanie modelu na obserwacjach z roku 2024 w celu prognozowania zdarzeń z roku 2018 zaburzyłoby ocenę właściwej mocy generalizacyjnej rozwiązania.

W celu zapewnienia miarodajnej oceny i wiernej symulacji warunków wdrożeniowych, wprowadzono twardy, chronologiczny podział zbioru 162 390 epizodów:
- **Zbiór treningowy (Train)**: Obserwacje historyczne z lat 2013–2021. Pula obejmująca 126 759 rekordów połączonej populacji psów i kotów.
- **Zbiór walidacyjny (Validation)**: Obserwacje z lat 2022–2023, liczące 21 811 epizodów. Przeznaczone do selekcji optymalnych modeli oraz punktów odcięcia.
- **Zbiór testowy (Test)**: Obserwacje wykraczające czasowo w lata 2024–2025, obejmujące 13 820 epizodów. Zbiór ten posłużył wyłącznie do jednokrotnej, ostatecznej oceny najlepszych estymatorów. Algorytmy optymalizacyjne nie posiadały do niego dostępu na żadnym etapie treningu.

Tego rodzaju strategia gwarantuje właściwą weryfikację odporności modeli na deaktualizację wzorców (ang. *concept drift*), co ma szczególne znaczenie w środowisku naznaczonym strukturalnymi pęknięciami (np. zmianami wynikającymi z załamania rynku po pandemii COVID-19).

## 4.4. Wykorzystane algorytmy predykcyjne

Proces ewaluacji uwzględnia szerokie spektrum architektur algorytmicznych, począwszy od prymitywnych modeli odniesienia, aż po nieliniowe estymatory zespołowe. Umożliwia to zbadanie stopnia zwrotu z inwestycji w narastającą złożoność obliczeniową.

**Modele bazowe (Baseline):**
- **Modele odniesienia (Dummy):** Algorytmy przewidujące niezmiennie dominantę w przypadku klasyfikacji oraz stałą medianę populacyjną dla regresji. Wyznaczają one minimalny pułap błędu obciążającego naiwne podejście deterministyczne.
- **Modele parametryczne liniowe:** Regresja logistyczna dla przestrzeni binarnej oraz regresja grzbietowa (Ridge) z regularyzacją L2 dla wartości ciągłych. Rozwiązania te oferują stabilność numeryczną oraz transparentność estymacji wag.

**Modele zaawansowane (Drzewa decyzyjne i Boosting):**
- **Lasy Losowe (Random Forest):** Podejście zespołowe oparte na metodzie *baggingu*. Gwarantuje relatywną odporność na przeuczenie oraz przechwytuje lokalne nieliniowości, stanowiąc wiodący standard ewaluacyjny dla danych stabelaryzowanych.
- **Histogram-based Gradient Boosting:** Adaptacja algorytmu sekwencyjnego uczenia błędów, wzorowana technicznie na rozwiązaniach typu LightGBM. Wykorzystanie dyskretyzacji (kubełkowania) atrybutów numerycznych gwarantuje redukcję złożoności czasowej na zbiorach osiągających rząd 100 tysięcy próbek.
- **CatBoost:** Specjalistyczna architektura z rodziny wzmacniania gradientowego (ang. *Gradient Boosting*). Ze względu na implementację zoptymalizowanego mechanizmu kodowania wartości kategorycznych (tzw. *Ordered Target Encoding*), algorytm doskonale sprawdza się w dziedzinie, która zdominowana jest przez setki nominalnych etykiet ras oraz umaszczeń, eliminując negatywne konsekwencje wymuszonego mnożenia rzadkich wektorów reprezentacją typu *one-hot*.

Z uwagi na ustrukturyzowany, stabelaryzowany charakter zebranych zbiorów z ekstremalnie zróżnicowaną kardynalnością predyktorów tekstowych, algorytmy iteracyjnego uczenia błędów (CatBoost, HistGradientBoosting) wyznaczają obecnie granicę możliwości analitycznych (*state-of-the-art*). Odstąpiono od implementacji głębokich sieci neuronowych (*Deep Learning*), których zastosowanie dla tego specyficznego formatu danych na ogół nie koreluje z przyrostem wartości AUC, obniżając jednocześnie właściwości eksplanacyjne systemu.

## 4.5. Metryki oceny

Z uwagi na specyfikę domenową problemu, proces weryfikacji rezultatów rozszerzono o dedykowane miary statystyczne.

**Kwantyfikacja zadania klasyfikacji:**
- **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**: Miara odporna na niewielkie zaburzenia balansu klas. Oblicza ogólną zdolność separacji rozkładów predykcji negatywnych i pozytywnych (gdzie $1.0$ implikuje separację idealną, a $0.5$ wskazuje na losowość estymatora).
- **AUC-PR (Area Under the Precision-Recall Curve)**: Miara uwydatniająca skuteczność predykcyjną w identyfikacji klasy istotnej, ignorując obfitość klasy negatywnej (prawdziwie ujemnej).
- **Metryki punktowe (F1-Score, Precyzja, Czułość)**: Wynikające z dychotomizacji prognozowanych prawdopodobieństw względem wybranego na zbiorze walidacyjnym punktu odcięcia (threshold). Pozwalają na ocenę potencjalnej decyzyjności wdrożonego modelu w warunkach ostrych kryteriów kategoryzujących.
- Weryfikowano również kalibrację prawdopodobieństw modelu, aby zapewnić adekwatność wyników w skali probabilistycznej.

**Kwantyfikacja zadania regresji:**
- **MAE (Mean Absolute Error)**: Średni błąd bezwzględny, wyrażony w dniach operacyjnych. Stanowi główną metrykę z racji liniowego karania za uchybienia, bez przyznawania asymetrycznej potęgi nielicznym ekstremalnym wartościom odstającym (np. psom pozostającym w ośrodku przez 4 lata).
- **MedAE (Median Absolute Error)**: Mediana z wektora błędów bezwzględnych. Skrajnie oporna na wpływ uchybień marginalnych. Oznacza centralny, typowy poziom pomyłki dla przeciętnego ujętego w analizie epizodu.
- **RMSE (Root Mean Square Error)**: Pierwiastek obłędu średniokwadratowego. Wykorzystany jako wsparcie ewaluacyjne w celu ujęcia skali dużych odchyleń w prognozach w relacji do metryki MAE.

Oceny dopełnia agregacja błędów w wybranych przekrojach, co pozwoli na wskazanie typów zwierząt (podgrup referencyjnych), których profil behawioralno-rozwojowy wymyka się statystycznej przewidywalności.

## 4.6. Interpretowalność procedur analitycznych (SHAP)

Wdrożenie technologii uczenia maszynowego na gruncie prognozowania obrotu podmiotami żywymi bezwzględnie wymaga wysokiego stopnia rygoru eksplanacyjnego. Decyzje przypisujące psu bądź kotu wirtualne prawdopodobieństwo długiej odsiadki w placówce nie mogą pozostawać zaciemnioną, abstrakcyjną formułą. Metodologią obraną do analizy zachowań modeli są wartości **SHAP** (ang. *SHapley Additive exPlanations*).

Rozwiązanie to, wywodzące się z kooperatywnej teorii gier, zapewnia dokładny i matematycznie konsystentny dekompozycyjny przypis odpowiedzialności poszczególnym atrybutom w kreowaniu przewidywań przez zawiłe struktury drzew decyzyjnych.
1. **Analiza rozkładów globalnych (SHAP Summary):** Zbiór punktów obrazujący zagregowany, kierunkowy wpływ cech dla setek obserwacji jednocześnie. Podważa ograniczenia tradycyjnego, bezkierunkowego współczynnika istoty (np. zysku informacyjnego Gini), pokazując explicite, w jaki sposób np. rosnący wiek koreluje ze spadkiem szansy adopcyjnej z uwzględnieniem oddziaływania innych atrybutów.
2. **Rekonstrukcja logiki lokalnej:** Ewaluacja izolowanego przebiegu dla wybranej obserwacji jednostkowej. Umożliwia przedstawienie wpływu cech jako sił addytywnych – bazy odniesienia, sił zmniejszających prognozę oraz sił wyciągających ją ku skrajnym prawdopodobieństwom.
3. **Analiza Ablacyjna Rodzin Cech:** Nowatorskie ujęcie problemu hiper-kardynalności zaimplementowane w architekturze pracy. Mechanizm dekomponuje całościowy wkład w podziale na semantyczne super-zbiory: tzw. Rodzinę Wieku, Rodzinę Wyglądu, Rodzinę Okoliczności. Umożliwia to testowanie wrażliwości predykcji poprzez hipotetyczną sterylizację całych gałęzi informacyjnych (np. poprzez zmuszenie modelu do ignorowania aspektów wizualnych w ocenie szansy znalezienia domu).

## 4.7. Protokół badawczy i parametryzacja

Zapewnienie odtwarzalności badawczej i ścisłej rozdzielności faz nauki i prezentacji stanowiły oś struktury eksperymentu.
1. **Zabezpieczenie przed chaosem entropijnym**: We wszystkich warstwach o charakterze niedeterministycznym zastosowano zunifikowane ziarno generatora liczb pseudolosowych (`random_state=42`).
2. **Modularność i separacja kompetencji**: Środowisko podzielono na ściśle odizolowane komponenty: proces treningowy wywoływany w konsoli deweloperskiej i generujący niezmienne struktury danych na dysku fizycznym.
3. **Trwałość artefaktów obliczeniowych**: Uzyskane modele utrwalano na twardym dysku maszyny (`.joblib`), a finalne oceny precyzji i wagi wpływu agregowano do surowych plików ewidencyjnych `.csv`. Powstała aplikacja analityczna i graficzna posiada uprawnienia do procesów czysto konsumpcyjnych – uniemożliwiono w niej retrenowanie czy aktualizację wartości predykcyjnych w czasie rzeczywistym.

## 4.8. Ramy eksploatacyjne modeli

Podjęte rygorystyczne zabiegi walidacyjne nie anulują inherentnych defektów samego przedmiotu modelowania statystycznego.

Technicznym limitem badawczym jest absolutna degradacja użyteczności modelu dla instancji operacyjnych objętych zjawiskiem aktywnego cenzurowania prawostronnego, co wymusiło ewaluację systemową wykluczającą zjawiska wciąż poszukujące prawomocnego zakończenia. 

W wymiarze etycznym i socjologicznym podkreśla się z kolei fakt, że parametry predykcyjne dla ras obarczonych złą sławą (m.in. pitbule) nie odzwierciedlają defektów owego przedmiotu (uwarunkowań genetycznych psów decydujących o rzadszych i późniejszych wyjściach). Obrazują precyzyjnie asocjację warunków percepcyjnych i strukturalnych amerykańskiego rynku na obszarze Austin w historycznym przedziale dekady. Traktowanie tych oszacowań jako prawa uniwersalnego byłoby nadużyciem epistemologicznym, a przeniesienie tak skonstruowanych estymatorów do schroniska kierującego się reżimem *kill-shelter* mogłoby spotkać się z dramatycznym odrzuceniem trafności bez uprzedniej, głębokiej kalibracji lokalnej.

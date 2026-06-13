# 3. Dane i przygotowanie zbioru badawczego

W niniejszym rozdziale opisano proces przygotowania zbioru danych przeznaczonych do analizy statystycznej. Jednostką obserwacji w badaniu jest epizod pobytu zwierzęcia w schronisku, czyli pojedyncze przyjęcie, przypisany mu wynik kończący pobyt oraz zbiór cech znanych w momencie przyjęcia (cechy *intake-time-only*). Przedstawiono etapy transformacji danych surowych do postaci analitycznej.

## 3.1. Źródło danych i zakres analizy

Materiał empiryczny pochodzi ze schroniska Austin Animal Center (AAC) i został pobrany z bazy Austin Open Data. Baza składa się z dwóch tabel: rejestru przyjęć zwierząt (intakes) oraz rejestru wyników kończących pobyt (outcomes). Opcjonalnie dane wzbogacono o zmienne kontekstowe (np. pogodę).

Zakres czasowy badania obejmuje przyjęcia od 1 października 2013 roku do 4 maja 2025 roku. Analizę zawężono do psów i kotów, ponieważ stanowią one główną część populacji schroniskowej. Obecność gatunków rzadkich (np. dzikich zwierząt) mogłaby obniżyć stabilność estymacji.

Wykorzystane dane mają charakter obserwacyjny. Pochodzą one z rutynowej dokumentacji schroniska, a nie z kontrolowanego eksperymentu. Z tego powodu analiza dotyczy wyłącznie relacji statystycznych pomiędzy atrybutami zwierzęcia przy przyjęciu a wynikiem pobytu. Nie wnioskuje się o związkach przyczynowo-skutkowych.

## 3.2. Charakterystyka zbiorów wejściowych

Zbiór łączy dane o przyjęciu z danymi o wyniku końcowym. Ich rozdzielenie na etapie modelowania pozwala zapobiec wykorzystywaniu informacji z przyszłości (wyciek danych).

**Tabela 3.1.** Porównanie dwóch podstawowych zbiorów wejściowych

| Warstwa danych | Przykładowe pola kluczowe | Rola w procesie |
|---|---|---|
| Rejestr przyjęć | `animal_id`, `animal_type`, `intake_datetime`, `intake_type`, `intake_condition`, `sex_upon_intake`, `age_upon_intake`, `breed`, `color` | Źródło atrybutów wejściowych ($X$) |
| Rejestr wyników | `animal_id`, `animal_type`, `outcome_datetime`, `outcome_type`, `outcome_subtype`, `sex_upon_outcome` | Źródło zmiennej docelowej ($y$) |

Metadane generowane na koniec pobytu (`outcome_type`, `outcome_datetime`) wyizolowano jako zmienne docelowe. Podział ten utrwalono w formacie oddzielnych plików konfiguracyjnych JSON dla cech i targetów.

## 3.3. Standaryzacja i czyszczenie danych

Czyszczenie danych przeprowadzono ostrożnie, bez sztucznej imputacji braków. Nazwy zmiennych zapisano w formacie `snake_case`. Daty ujednolicono do lokalnej strefy czasowej schroniska, konwertując ciągi znaków do formatu `datetime64`.

Sprawdzono kompletność kluczowych atrybutów (`animal_id`, `animal_type`, `intake_datetime`, `outcome_datetime`, `outcome_type`). Obserwacje z brakami w tych polach usunięto. Dokładne duplikaty rekordów zostały odfiltrowane.

**Tabela 3.2.** Atrycja danych w procesie budowy finalnego zbioru modelowego

| Zbiór danych na danym etapie | Liczba wierszy | Liczba usuniętych |
|---|---:|---:|
| Zbiór źródłowy wejść (`intakes.csv`) | 173 812 | — |
| Zbiór źródłowy wyjść (`outcomes.csv`) | 173 775 | — |
| Standaryzacja i filtracja wejść | 163 901 | 9 911 |
| Standaryzacja i filtracja wyjść | 163 880 | 9 895 |
| Dopasowane epizody z targetem | 162 390 | 1 511 |
| Finalny zbiór modelowy | 162 390 | 0 |

Większość usuniętych rekordów to zwierzęta inne niż psy i koty. Usunięto również 1 511 przyjęć pozbawionych chronologicznie poprawnego wyniku końcowego (cenzorowanie). Brakujące dane w atrybutach predykcyjnych dotyczyły pojedynczych przypadków (9 braków wieku, 1 brak płci).

## 3.4. Konstrukcja epizodów i dopasowanie rekordów

Jednostką analizy jest epizod pobytu. Ponieważ to samo zwierzę (`animal_id`) może zostać kilkukrotnie zarejestrowane w schronisku, w celu odtworzenia historii pobytu każde przyjęcie połączono z najbliższym w czasie, niewykorzystanym wynikiem (metoda najbliższego wolnego zdarzenia w czasie).

Taka procedura zapobiega powstaniu ujemnych czasów pobytu oraz wielokrotnemu użyciu tego samego wyniku dla różnych przyjęć. Audyt zbioru potwierdził poprawność łączenia – nie odnotowano epizodów nakładających się w czasie.

## 3.5. Zmienne docelowe

Zdefiniowano dwie główne zmienne docelowe do późniejszego modelowania:

**Tabela 3.3.** Formalne definicje zmiennych docelowych

| Zmienna | Definicja | Interpretacja |
|---|---|---|
| `classification_target` | `1` dla "Adoption", `0` dla innych | Binarny wynik adopcji |
| `regression_target_days` | $(outcome\_time - intake\_time) \div 86400$ | Całkowita długość pobytu (dni) |
| `days_to_adoption` | Wartość `regression_target_days` dla adoptowanych | Długość pobytu do adopcji |

Adopcją zakończyło się 50,87% (82 609) uwzględnionych epizodów. Pozostałe 49,13% obejmuje powroty do właściciela, transfery, eutanazję i zgony. Czas do dowolnego zdarzenia potraktowano jako klasyczny problem regresji ze względu na bezpośrednie przełożenie długości zajmowania boksu na koszty operacyjne placówki.

## 3.6. Konstrukcja cech wejściowych

Cechy wejściowe zbudowano wyłącznie z informacji dostępnych w chwili przyjęcia. 

**Tabela 3.4.** Grupy cech wejściowych

| Grupa | Zmienne |
|---|---|
| Cechy ogólne | `animal_type`, `sex_upon_intake`, `intake_condition` |
| Wiek | `age_days`, `age_years`, `age_group` (np. baby, senior) |
| Rasa | `breed`, `primary_breed`, `simplified_breed_group` |
| Umaszczenie | `color`, `primary_color`, `simplified_color_group` |
| Kontekst przyjęcia | `intake_season`, `intake_type`, `covid_period` |

W celu redukcji liczby kategorii, rasy i umaszczenia pogrupowano w szersze kategorie (np. `pit_bull_type`, `terrier_type`, `domestic_cat`). Oryginalna zmienna z rasą posiadała 2 761 unikalnych wartości, z czego większość występowała marginalnie rzadko.

Na podstawie daty przyjęcia zdefiniowano okresy związane z pandemią: przed COVID-19 (do 1 marca 2020 r.), w trakcie pandemii (marzec 2020 – grudzień 2021) oraz po pandemii. Wiek przekształcono z wartości tekstowych (np. "2 years") do postaci w pełni numerycznej (w dniach) oraz kategorii wiekowych.

Cechy wejściowe zostały dodatkowo przebadane pod kątem zjawiska współliniowości (multikolinearności). Zastosowano analizę wskaźnika VIF (*Variance Inflation Factor*), aby upewnić się, że modele oparte na założeniach addytywnych (takie jak regresja logistyczna) nie zostaną zniekształcone przez redundancję informacji (np. skorelowanie wieku w dniach z wiekiem w latach). Zmienne o krytycznie wysokim współczynniku VIF zostały usunięte na wczesnym etapie selekcji cech.

## 3.7. Charakterystyka opisowa zbioru

Zbiór 162 390 pobytów pozwala na przedstawienie podstawowych statystyk opisowych. Podstawowe wskaźniki zmieniały się w czasie:
* **Przed COVID-19 (108 812 epizodów):** Wskaźnik adopcji wynosił 46,3%, przy medianie całkowitej długości pobytu (dla każdego wyniku) równej 5,25 dnia.
* **COVID-19 (17 947 epizodów):** Wskaźnik adopcji wzrósł do 57,2%, a mediana pobytu do 8,01 dnia.
* **Post COVID-19 (35 631 epizodów):** Wskaźnik adopcji wyniósł 61,5%, a mediana pobytu 9,83 dnia.

Zbiega się to ze zmianą struktury przyjmowanych zwierząt. Odsetek psów spadł z 60,45% w okresie przed-pandemicznym do 50,64%. Z kolei odsetek młodych zwierząt (`baby`) wzrósł z 44,11% do 54,34%.

Psy i koty różnią się pod względem dynamiki adopcji. Ogólny wskaźnik adopcji był zbliżony (psy: 50,28%, koty: 51,68%). Jednak wśród zwierząt adoptowanych, do 30 dni schronisko opuszcza 73,6% psów i zaledwie 54,5% kotów.

Prawdopodobieństwo adopcji maleje wraz z wiekiem zwierzęcia. Grupa najmłodsza (`baby`) charakteryzuje się wskaźnikiem adopcji na poziomie 59,7%, młodzież (`young`) – 47,5%, dorosłe (`adult`) – 39,6%, a starsze (`senior`) – 31,0%.

Sposób przyjęcia również wykazuje dużą zmienność wyników. Wśród 116 393 zwierząt znalezionych na ulicy (`Stray`) adopcją zakończyło się 48,7% pobytów, natomiast z 34 142 zrzeczeń właścicielskich (`Owner Surrender`) wskaźnik wyniósł 67,1%.

## 3.8. Ograniczenia danych

Wykorzystany zbiór danych posiada kilka ograniczeń.
Po pierwsze, pochodzi on z placówki no-kill (ponad 90% przeżywalności), która nie stosuje rutynowej eutanazji. Wyniki mogą różnić się od schronisk o innej specyfice.
Po drugie, wpisy dotyczące rasy i umaszczenia są deklaracją pracownika przy przyjęciu, a nie wynikiem badań genetycznych. Etykiety typu `pit_bull_type` oddają administracyjny sposób kategoryzacji psów i ludzką percepcję ich wyglądu.
Ograniczeniem technicznym było wykluczenie pobytów bez prawomocnie zakończonego wyniku, co uwarunkowane jest wymogiem posiadania etykiety docelowej w nadzorowanym uczeniu maszynowym.

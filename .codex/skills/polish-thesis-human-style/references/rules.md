# Szczegółowe reguły redakcji polskiej pracy naukowej

## Spis treści

1. Hierarchia decyzji
2. Wierność faktom
3. Cytowania i źródła
4. Wyniki ilościowe
5. Metodologia i wnioskowanie
6. Polski styl naukowy
7. Sygnały tekstu szablonowego
8. Struktura rozdziału i akapitu
9. LaTeX
10. Lista kontrolna

## 1. Hierarchia decyzji

Przy konflikcie stosować kolejność:

1. prawdziwość i zgodność ze źródłem;
2. zachowanie znaczenia i stopnia pewności;
3. zgodność z metodologią projektu;
4. kompletność informacji;
5. czytelność;
6. naturalność stylu;
7. zwięzłość.

Nie poświęcać pozycji 1–4 dla pozycji 5–7.

## 2. Wierność faktom

### Zakazane operacje

- dodawanie prawdopodobnych faktów, przykładów lub mechanizmów;
- usuwanie faktu, ponieważ wystąpił wcześniej, jeśli pełni inną funkcję;
- zmiana zakresu: „w próbie” na „w populacji”, „w 2024 r.” na „obecnie”;
- zmiana ilości: „większość” na „niemal wszystkie”;
- zmiana kierunku: „niższy” na „odmienny” albo „związany” na „korzystny”;
- zmiana modalności: „może wskazywać” na „wskazuje”;
- rekonstruowanie brakującej wartości na podstawie sąsiednich danych;
- naprawianie pozornej sprzeczności bez sprawdzenia producenta danych.

### Dozwolone operacje

- korekta fleksji, składni, interpunkcji i szyku;
- usunięcie pustego wartościowania, jeżeli nie niesie faktu;
- rozdzielenie długiego zdania przy zachowaniu relacji logicznych;
- połączenie zdań tylko wtedy, gdy nie zmienia to zakresu informacji i cytowań;
- konsekwentne użycie ustalonego terminu;
- zmiana formatu liczby zgodnie z zasadami projektu, bez zmiany wartości.

### Test minimalnej pary

Zapytać: czy czytelnik mógłby na podstawie nowej wersji udzielić innej
odpowiedzi na pytanie o dane, metodę, zakres lub pewność wniosku? Jeśli tak,
redakcja zmieniła treść.

## 3. Cytowania i źródła

- Zachować wszystkie cytowania, nawet gdy kilka występuje obok siebie.
- Nie tworzyć nowych wpisów bibliograficznych bez zweryfikowanego źródła.
- Nie podawać numeru strony, DOI, ISBN ani URL z pamięci.
- Nie rozszerzać twierdzenia poza zakres wspierany przez cytowane źródło.
- Nie przypisywać autorowi interpretacji redaktora.
- Cytowanie po akapicie nie musi wspierać każdego zdania; ustalić jego zakres
  przed przebudową akapitu.
- Przy tłumaczeniu nie tłumaczyć nazw własnych, tytułów lub terminów technicznych
  bez sprawdzenia przyjętej polskiej formy.
- Brak dostępu do źródła oznaczyć jako ograniczenie. Nie udawać weryfikacji.

## 4. Wyniki ilościowe

Chronić jako jeden pakiet:

- wartość;
- metrykę;
- model lub grupę;
- zbiór i okres oceny;
- jednostkę;
- kierunek optymalizacji;
- niepewność lub zastrzeżenie.

Przykład pakietu: „CatBoost uzyskał ROC-AUC 0,841 na zbiorze testowym z lat
2024–2025”. Nie wolno pozostawić samego „model osiągnął 0,841”, jeśli nazwa
metryki albo okres są potrzebne do interpretacji.

Nie porównywać metryk o innym znaczeniu. Nie nazywać różnicy istotną
statystycznie bez odpowiedniego testu. „Najlepszy” zawsze wymaga wskazania
kryterium i badanego zestawu kandydatów.

## 5. Metodologia i wnioskowanie

Rozróżniać:

- **opis**: co zaobserwowano w danych;
- **predykcję**: jak model przewiduje wynik dla danych wejściowych;
- **interpretację modelu**: które cechy wiążą się z predykcją modelu;
- **wnioskowanie przyczynowe**: jaki efekt wywołuje interwencja.

Nie przechodzić między tymi poziomami bez podstawy metodologicznej.

Słowa wymagające dowodu przyczynowego: „powoduje”, „prowadzi do”, „wpływa na”,
„skutkuje”, „dzięki temu”, „w wyniku”. W badaniu obserwacyjnym preferować
precyzyjne formy: „wiązało się z”, „zaobserwowano”, „model przypisywał”, „wynik
był niższy w analizowanej próbie”.

Zachowywać rozdzielenie zbioru treningowego, walidacyjnego i testowego. Nie
opisywać wyboru dokonanego na walidacji jako wyniku uzyskanego na teście.
Nie przedstawiać losowego podziału awaryjnego jako ewaluacji chronologicznej.

## 6. Polski styl naukowy

### Preferować

- zdania, których orzeczenie mówi, co zrobiono lub zaobserwowano;
- konkret przed oceną;
- jeden ustalony termin dla jednego pojęcia;
- jawny podmiot, gdy autor działania ma znaczenie;
- stronę bierną, gdy procedura lub obiekt są ważniejsze od wykonawcy;
- krótkie przejścia logiczne wynikające z relacji treści;
- akapity różnej długości, jeśli wynika to z materiału.

### Ograniczać

- „warto zauważyć”, „należy podkreślić”, „co istotne”;
- „kluczowy”, „fundamentalny”, „niezwykle ważny”, „kompleksowy” bez kryterium;
- „dynamiczny”, „wielowymiarowy”, „holistyczny”, „innowacyjny” bez definicji;
- „w dzisiejszych czasach”, „w dobie”, „we współczesnym świecie”;
- „niniejszy rozdział ma na celu” powtarzane w każdej sekcji;
- „wyniki jednoznacznie dowodzą”, gdy analiza daje słabszą podstawę;
- seryjne imiesłowy dopowiadające ogólną korzyść lub znaczenie;
- nominalizacje, jeśli prosty czasownik jest precyzyjniejszy;
- przesadne unikanie powtórzeń terminologicznych.

Nie zakazywać wszystkich przysłówków, strony biernej, zdań długich ani
wyliczeń. W polskim tekście akademickim mogą być uzasadnione.

## 7. Sygnały tekstu szablonowego

Poniższe zjawiska są sygnałami do sprawdzenia, nie dowodem użycia AI:

### Nadawanie sztucznej doniosłości

Zdanie dopowiada, że wynik „podkreśla znaczenie”, „odzwierciedla szerszy trend”
lub „stanowi ważny krok”, lecz nie wskazuje wyniku, mechanizmu ani źródła.
Usunąć pustą ocenę albo zastąpić ją konkretną konsekwencją obecną w materiale.

### Powierzchowna interpretacja

Do faktu dołączono ogólną frazę o „cennych wnioskach”, „złożoności problemu”,
„potrzebie dalszych działań” lub „praktycznym znaczeniu”. Zachować tylko
interpretację wynikającą z danych, metody lub cytowanego źródła.

### Mechaniczne kontrasty

Konstrukcje „nie tylko..., ale również...” i „nie jest to X, lecz Y” bywają
użyte bez rzeczywistego kontrastu. Zapisać relację bez retorycznej ramy. Nie
usuwać konstrukcji, jeśli negacja jest merytorycznie potrzebna.

### Reguła trzech

Model często tworzy wyliczenia dokładnie trzech abstrakcyjnych korzyści,
wyzwań lub aspektów. Liczba elementów ma wynikać z materiału. Nie redukować
prawdziwej listy tylko dlatego, że ma trzy pozycje.

### Wymuszona różnorodność leksykalna

Zmiana „modelu” na „algorytm”, „narzędzie”, „rozwiązanie” i „system” może
zacierać różnice. Powtórzenie terminu naukowego jest lepsze niż fałszywy
synonim.

### Symetryczna, podręcznikowa struktura

Każda sekcja ma identyczny wstęp, trzy punkty, podsumowanie i deklarację
znaczenia. Zmieniać strukturę tylko wtedy, gdy nie usuwa to wymaganych części
argumentu.

### Metatekst i komunikacja do użytkownika

Usuwać z pracy ślady rozmowy: „oto poprawiona wersja”, „mam nadzieję, że”,
„można dostosować”, „jako model”. Nie usuwać uzasadnionych zapowiedzi struktury
rozdziału.

## 8. Struktura rozdziału i akapitu

### Akapit wynikowy

Zwykle zawiera:

1. pytanie lub porównanie;
2. wynik wraz z kontekstem pomiaru;
3. ostrożną interpretację;
4. ograniczenie, jeśli wpływa na odczyt wyniku.

Nie wymuszać wszystkich czterech elementów w każdym akapicie. Nie kończyć
automatycznie zdaniem o znaczeniu wyniku.

### Akapit metodologiczny

Wskazać decyzję, jej uzasadnienie, sposób wykonania i konsekwencję dla
interpretacji. Nie przypisywać metodzie zalet, których nie wykazano.

### Wstęp i zakończenie

Wstęp ma określić problem, zakres i cel. Zakończenie ma syntetyzować odpowiedzi
na pytania badawcze oraz ograniczenia. Nie dodawać nowych wyników lub źródeł w
zakończeniu.

## 9. LaTeX

- Nie zmieniać argumentów `\cite`, `\ref`, `\label`, `\input`, `\includegraphics`
  ani własnych makr bez wyraźnej potrzeby.
- Zachować znaki specjalne i escapowanie: `\%`, `\_`, `\&`, `~`.
- Nie zamieniać myślników w zakresach dat, jeśli projekt określa zapis LaTeX.
- Nie usuwać niewidocznie znaczących nawiasów klamrowych.
- Nie przenosić `\label` w sposób zmieniający numerowaną strukturę.
- Po zmianach obejmujących polecenia lub podział akapitów uruchomić kompilację
  albo odpowiedni test, jeśli środowisko na to pozwala.

## 10. Lista kontrolna

### Treść

- [ ] Nie dodano żadnego faktu ani interpretacji.
- [ ] Nie pominięto żadnego faktu, warunku ani zastrzeżenia.
- [ ] Liczby, daty, jednostki i kierunki relacji są zgodne.
- [ ] Stopień pewności każdego twierdzenia pozostał ten sam.
- [ ] Opis i predykcja nie zostały przedstawione jako przyczynowość.
- [ ] Terminologia jest zgodna z dokumentacją projektu.

### Źródła

- [ ] Każde cytowanie zachowano.
- [ ] Zakres cytowania nie został poszerzony.
- [ ] Nie utworzono źródła ani danych bibliograficznych z pamięci.
- [ ] Nie zadeklarowano weryfikacji źródła, którego nie odczytano.

### Styl

- [ ] Usunięto puste wartościowanie i ogólne dopowiedzenia.
- [ ] Każde zdanie wnosi informację albo potrzebną relację logiczną.
- [ ] Terminy nie są zastępowane wymuszonymi synonimami.
- [ ] Rytm wynika z treści, a nie z mechanicznego wzorca.
- [ ] Tekst pozostaje formalny, precyzyjny i naturalny po polsku.

### Format

- [ ] Składnia LaTeX pozostała poprawna.
- [ ] Etykiety, odsyłacze, cytowania i makra są nienaruszone.
- [ ] Zastosowano projektowe zasady zapisu liczb i terminów.

## Podstawa heurystyk

Katalog sygnałów tekstu szablonowego adaptuje obserwacje z
[Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
do polskiego tekstu akademickiego. Strona ma charakter opisowy, nie normatywny:
pojedyncza cecha nie dowodzi użycia AI, a celem redakcji nie jest ukrywanie
autorstwa. Reguły tego skilla koncentrują się na jakości argumentu, jawności
źródeł oraz zachowaniu treści.

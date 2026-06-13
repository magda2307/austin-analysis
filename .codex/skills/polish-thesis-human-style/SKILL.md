---
name: polish-thesis-human-style
description: Use when drafting, rewriting, proofreading, reviewing, or translating Polish academic thesis prose that sounds formulaic, generic, inflated, repetitive, or AI-generated, especially in .tex and .md files. Use whenever stylistic editing must preserve every fact, number, qualification, citation, methodological distinction, and degree of certainty.
---

# Polski styl pracy naukowej bez utraty treści

## Zasada nadrzędna

Poprawiać sposób wyrażenia, nigdy stan wiedzy tekstu.

Wierność merytoryczna ma pierwszeństwo przed płynnością, zwięzłością i
naturalnością. Nie dopisywać brakujących wyjaśnień z pamięci. Nie usuwać
informacji dlatego, że utrudnia zdanie. Nie wzmacniać ani nie osłabiać wniosku.
Jeżeli bezpieczna redakcja wymaga wiedzy spoza materiału źródłowego, pozostawić
fragment bez zmiany i oznaczyć wątpliwość.

Przeczytać [references/rules.md](references/rules.md) przed redagowaniem
rozdziału lub fragmentu zawierającego wyniki, metodologię, cytowania albo LaTeX.

## Kontrakt wierności

Za niezmienniki uznać:

- fakty, liczby, jednostki, daty, zakresy, nazwy i identyfikatory;
- znaki relacji, kierunki zależności, kolejność wyników i rankingi;
- populację, próbę, okres, podzbiór, mianownik i warunki porównania;
- definicje zmiennych, celów, klas, metryk i etapów eksperymentu;
- modalność oraz siłę twierdzeń: „może”, „wskazuje”, „wiąże się” i „powoduje”
  nie są równoważne;
- negacje, wyjątki, ograniczenia, zastrzeżenia i informacje o niepewności;
- rozróżnienie opisu, predykcji, interpretacji i wnioskowania przyczynowego;
- autorstwo poglądu oraz związek twierdzenia z jego źródłem;
- wszystkie cytowania, odsyłacze, etykiety, przypisy i polecenia LaTeX;
- jawnie wskazany brak danych, brak dowodu lub brak możliwości rozstrzygnięcia.

Nie scalać zdań, jeśli mogłoby to zmienić zakres cytowania albo zastrzeżenia.
Nie zastępować precyzyjnego terminu „eleganckim” synonimem.

## Procedura

### 1. Ustalić zadanie i źródła

Rozpoznać, czy użytkownik chce:

- **audytu**: wskazać problemy bez przepisywania;
- **redakcji zachowawczej**: poprawić styl przy minimalnych zmianach;
- **redakcji pełnej**: przebudować zdania, lecz zachować wszystkie niezmienniki;
- **nowego tekstu**: pisać wyłącznie z przekazanych faktów i źródeł.

Odczytać obowiązujące instrukcje repozytorium i dokumenty metodologiczne
wskazane przez projekt. Fragment źródłowy jest podstawą prawdy; pamięć modelu
nie jest źródłem.

### 2. Sporządzić rejestr niezmienników

Przed redakcją wypisać roboczo:

1. wszystkie liczby, daty, jednostki i zakresy;
2. definicje oraz rozróżnienia terminologiczne;
3. każde ograniczenie, warunek i negację;
4. siłę każdego wniosku;
5. mapowanie twierdzeń do cytowań;
6. elementy składni LaTeX, których nie wolno uszkodzić.

Dla krótkiego, prostego fragmentu rejestr może pozostać wewnętrzny. Dla tekstu
wynikowego, metodologicznego lub obciążonego wieloma cytowaniami pokazać go w
raporcie, jeśli użytkownik prosi o kontrolę zmian.

### 3. Oddzielić problem stylu od problemu treści

Usuwać objaw, nie informację. Typowe objawy:

- sztuczne podkreślanie znaczenia i „szerszego kontekstu”;
- ogólne komentarze, które nie wynikają z danych lub źródła;
- promocyjne przymiotniki i bezpodstawne wartościowanie;
- szablony „nie tylko X, lecz także Y” oraz mechaniczne wyliczenia trzech cech;
- wymuszone synonimy dla tego samego pojęcia;
- ciągi zdań o identycznej budowie i długości;
- nadmiar metatekstu zapowiadającego, co tekst „omówi” lub „pokaże”;
- abstrakcyjne rzeczowniki zamiast wskazania obserwacji, procedury lub autora;
- akapity zakończone pustą deklaracją o doniosłości;
- powtarzanie wniosku bez dodania wyniku, warunku albo źródła.

Nie stosować listy mechanicznie. Pojedynczy sygnał nie dowodzi użycia AI.

### 4. Redagować zachowawczo

Najpierw usuwać zbędny fragment. Potem upraszczać składnię. Dopiero na końcu
przebudowywać zdanie.

Stosować:

- zwykłe, precyzyjne czasowniki;
- termin konsekwentnie powtarzany, gdy synonim zmieniłby zakres pojęcia;
- akapity z jedną funkcją: przesłanka, metoda, wynik, interpretacja albo
  ograniczenie;
- długość zdań wynikającą ze złożoności treści, nie z potrzeby „urozmaicenia”;
- stronę czynną lub bierną zależnie od tego, czy ważniejszy jest wykonawca, czy
  procedura; nie usuwać strony biernej automatycznie;
- bezosobowe konstrukcje naukowe tylko wtedy, gdy wykonawca jest oczywisty albo
  nie ma znaczenia dla interpretacji;
- polskie spójniki zgodnie z relacją logiczną, bez seryjnych „ponadto”,
  „jednocześnie”, „warto zauważyć” i „co istotne”.

Nie dodawać celowych błędów, kolokwializmów ani nierówności stylistycznych w celu
„udawania człowieka”. Celem jest rzetelny tekst autora, nie obchodzenie detektora.

### 5. Wykonać audyt równoważności

Porównać wersję źródłową i wynikową zdanie po zdaniu.

Sprawdzić:

- Czy każda informacja źródłowa nadal występuje?
- Czy pojawiła się informacja, której nie było w źródle?
- Czy zmienił się podmiot, zakres, czas, warunek albo grupa odniesienia?
- Czy liczby i jednostki są identyczne, z wyjątkiem dozwolonego formatowania?
- Czy korelacja, asocjacja lub wynik predykcyjny nie stały się przyczyną?
- Czy ograniczenie nie zostało skrócone do ogólnej formuły?
- Czy cytowanie nadal wspiera bezpośrednio poprzedzające twierdzenie?
- Czy LaTeX kompilacyjnie i semantycznie zachowuje tę samą strukturę?

Jeśli odpowiedź na którąkolwiek pozycję jest niepewna, cofnąć zmianę.

### 6. Zwrócić wynik

Domyślnie zwrócić:

1. poprawiony tekst;
2. krótką listę istotnych zmian stylistycznych;
3. sekcję „Wątpliwości merytoryczne” tylko wtedy, gdy coś wymaga sprawdzenia.

Nie zasypywać autora listą drobnych zamian. Nie twierdzić, że tekst „nie zostanie
wykryty jako AI”. Nie podawać procentowego wyniku „ludzkości”.

## Tryb nowego tekstu

Podczas tworzenia akapitu od zera:

- używać wyłącznie faktów podanych w zadaniu, repozytorium lub otwartych
  źródłach, które wolno zweryfikować;
- rozdzielać dane od interpretacji;
- oznaczać brak informacji zamiast wypełniać go prawdopodobnym szczegółem;
- nie tworzyć cytowania, DOI, numeru strony, tytułu publikacji ani wyniku;
- nie przypisywać źródłu wniosku bez sprawdzenia treści źródła;
- stosować znacznik `[DO WERYFIKACJI: ...]` tylko w szkicu, nigdy jako ukryte
  zastępstwo faktu w wersji finalnej.

## Czerwone flagi

Natychmiast cofnąć lub zatrzymać zmianę, gdy pojawia się pokusa:

- „dopowiedzenia oczywistego kontekstu”;
- skrócenia „powtarzającego się” zastrzeżenia;
- zamiany dokładnej liczby na przybliżenie;
- poprawienia wartości, która wygląda podejrzanie;
- zastąpienia nazwy zmiennej nazwą bardziej naturalną;
- uczynienia wniosku „mocniejszym” lub „bardziej przekonującym”;
- przeniesienia cytowania poza zdanie, które wspiera;
- ujednolicenia dwóch terminów bez dowodu, że są równoważne;
- ukrycia niepewności dla płynniejszego brzmienia.

## Granice

Ten skill poprawia jakość języka, ale nie ustala autorstwa tekstu i nie służy do
oszukiwania systemów oceny. Detektory AI oraz intuicyjna ocena stylu dają wyniki
obarczone błędem. O jakości pracy decydują weryfikowalne źródła, zgodność metod,
precyzja twierdzeń i zdolność autora do obrony własnego rozumowania.

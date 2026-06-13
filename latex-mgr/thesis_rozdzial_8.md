# 8. Podsumowanie i wnioski końcowe

Ostatni rozdział pracy stanowi syntezę przeprowadzonych badań, ewaluację stopnia realizacji założonych celów oraz nakreślenie obszarów wymagających dalszej eksploracji w przyszłości. 

Głównym celem niniejszej pracy było zbadanie możliwości wykorzystania technik uczenia maszynowego do przewidywania adopcyjności i czasu pobytu zwierząt (psów i kotów) w schronisku typu *no-kill* na podstawie danych dostępnych wyłącznie w momencie ich przyjęcia. Wyniki eksperymentów, oparte na analizie ponad 160 tysięcy epizodów z Austin Animal Center, potwierdzają wysoką skuteczność analityczną zastosowanych algorytmów z rodziny drzew decyzyjnych (*Gradient Boosting*). Zbudowane modele potrafią z dużą precyzją wyabstrahować ukryte wzorce decydujące o ostatecznym losie zwierzęcia.

## 8.1. Odpowiedzi na pytania badawcze (Hipotezy)

Podsumowanie statusu pięciu głównych hipotez sformułowanych we wstępie pracy przedstawia poniższa macierz ewaluacyjna:

**Tabela 8.1. Macierz ewaluacyjna hipotez badawczych**

| Hipoteza | Podsumowanie wyników z eksperymentów | Ostateczny status |
|---|---|---|
| **H1:** Okoliczności przyjęcia mają większy wpływ predykcyjny na wynik pobytu niż wrodzone cechy wyglądu zwierzęcia. | Modele (analiza ważności cech SHAP) wykazują, że kategoria przyjęcia (`Owner Surrender`, `Stray`, `Confiscate`) generuje silniejszy i bardziej jednoznaczny sygnał decyzyjny niż cechy umaszczenia. | **Zatwierdzona** |
| **H2:** Występuje istotna sezonowość w adopcjach, z miesiącami o wyraźnie niższej adopcyjności. | Mimo występowania różnic opisowych między miesiącami i porami roku, cechy sezonowe okazały się mieć drugorzędne znaczenie predykcyjne dla samego modelu uczenia maszynowego w porównaniu do wieku czy typu przyjęcia. | **Wsparta częściowo** (opisowo) |
| **H3:** Wiek zwierzęcia w momencie przyjęcia jest odwrotnie proporcjonalny do szansy na jego adopcję. | Hipoteza w pełni potwierdzona. Grupa wiekowa `Baby` ma wskaźnik adopcji rzędu blisko 60%, podczas gdy u zwierząt zaklasyfikowanych jako `Senior` spada on w okolice 31%. Wiek jest najważniejszym predyktorem w modelach globalnych. | **Zatwierdzona** |
| **H4:** Zwierzęta o ciemnym umaszczeniu przebywają w schronisku statystycznie dłużej ze względu na niższą adopcyjność. | Analiza opisowa wykazała nieznacznie niższe wskaźniki adopcji dla grupy *dark/black*. Jednak efekt ten na etapie wielowymiarowego uczenia maszynowego bywa często niwelowany przez cechy o wyższym priorytecie (wiek, wielkość rasy). Dowodzi to statystycznej korelacji, ale nie potwierdza strukturalnej dyskryminacji. | **Wsparta częściowo** (opisowo) |
| **H5:** Okres pandemii COVID-19 trwale zaburzył historyczne wzorce przepływu populacji schroniskowej. | Potwierdzono strukturalną zmianę. Pandemia zwiększyła sam odsetek zwierząt adoptowanych, ale drastycznie wydłużyła medianę długości pobytu z około 5 do blisko 10 dni, co zmieniło całkowitą dynamikę rotacji w placówce. | **Zatwierdzona** |

## 8.2. Ograniczenia badawcze i metodologiczne

Obiektywna ocena modeli predykcyjnych wymaga uczciwego wskazania ich limitacji. 

1. **Cenzorowanie prawostronne (Right-Censoring):** Do trenowania klasycznych algorytmów predykcyjnych (w szczególności regresji długości pobytu) konieczne było pominięcie epizodów, które nie posiadały wpisanego wyniku zakończenia w momencie zrzutu bazy danych. W efekcie modele mogły zostać nieznacznie odchylone w stronę faworyzowania krótkich, szybkich pobytów. Zjawisko to jest naturalną wadą odstąpienia od technik analizy przeżycia (Survival Analysis).
2. **Specyfika schronisk typu No-Kill:** Wszystkie zidentyfikowane wzorce odnoszą się do placówki działającej w środowisku o wysokiej świadomości społecznej (Teksas, miasto Austin), utrzymującej wskaźnik przeżywalności (tzw. *live release rate*) znacznie powyżej 90%. Z tego powodu transfer modelu z Austin do schroniska miejskiego w innej jurysdykcji o restrykcyjnej polityce eutanazji (tzw. *high-kill shelter*) skutkowałby całkowitą utratą właściwości kalibracyjnych.
3. **Brak zmiennych behawioralnych:** W wektorze wejściowym, w związku z założeniem predykcji tuż po przyjęciu (*intake-only*), nie zawarto oceny behawioralnej zwierzęcia. Zmienne takie jak reakcja na stres, agresja lękowa czy przyjazność, oceniane zazwyczaj w pierwszych 72 godzinach po przyjęciu, mogłyby znacząco podnieść skuteczność wariantu regresyjnego (MAE).

## 8.3. Kierunki dalszych badań (Future Work)

Rezultaty uzyskane w niniejszej pracy otwierają drogę do potencjalnych rozszerzeń o charakterze naukowym i wdrożeniowym:

* **Wdrożenie Głębokiej Analizy Przeżycia (Deep Survival Analysis):** Rozwiązaniem omijającym problem cenzorowania prawostronnego jest zastąpienie standardowej regresji sieciami neuronowymi przewidującymi funkcję hazardu w czasie (np. algorytmy *DeepSurv*). Pozwoliłoby to na ciągłe aktualizowanie prawdopodobieństwa adopcji z każdym kolejnym dniem pobytu zwierzęcia w boksie.
* **Przetwarzanie tekstu (NLP) uwag weterynaryjnych:** Wiele cennych, subtelnych sygnałów (np. "pies ma skłonność do ucieczek") jest przechowywanych w nieskompresowanych notatkach personelu medycznego. Wykorzystanie dużych modeli językowych (LLM) do ekstrakcji i kwantyfikacji tych danych wzbogaciłoby słownik cech kontekstowych.
* **Praktyczne wdrożenie operacyjne (Continuous Triaging):** Rozbudowanie zaprezentowanego w pracy narzędzia *Streamlit* do postaci zintegrowanego systemu klasy ERP, który łączyłby się bezpośrednio z systemami ewidencyjnymi schronisk (np. *Chameleon Software* lub *Shelterluv*), automatycznie oflagowując nowe, trudne adopcyjnie przypadki jeszcze przed porannym przeglądem dokonywanym przez weterynarza.

![Profile najwyższego ryzyka](file:///C:/Users/paula/Documents/mgr pjatk/reports/figures/vulnerable_profiles.png)
*[Rysunek 8.1. Zarys budowy behawioralnych profili ryzyka (vulnerable profiles) zidentyfikowanych w trakcie badania, które mogłyby stanowić fundament dla zautomatyzowanego systemu powiadomień operacyjnych]*

Wdrożenie systemów uczenia maszynowego w sektorze ratownictwa zwierząt nie zastąpi specjalistycznej wiedzy ludzkiego personelu. Niemniej jednak, jak udowodniono w tej pracy, stanowi ono znakomity mechanizm wspierający zarządzanie kryzysowe i dystrybucję ograniczonych zasobów. Skierowanie większej uwagi behawioralnej i promocyjnej na zwierzęta zidentyfikowane przez model jako wysoce narażone na długi pobyt może stanowić realny krok w stronę usprawnienia systemu współczesnej opieki nad bezdomnymi zwierzętami.

## 8.4. Podsumowanie końcowe i reprodukowalność

Przeprowadzona w niniejszej pracy analiza ponad 160 tysięcy epizodów schroniskowych dowiodła, że algorytmy uczenia maszynowego (w szczególności modele z rodziny *Gradient Boosting*) są w stanie skutecznie modelować losy zwierząt wyłącznie na podstawie początkowego profilu wejściowego. Zidentyfikowano kluczowe determinanty ryzyka, potwierdzając przewagę okoliczności przyjęcia (zrzeczenie, znalezisko) oraz wieku nad tradycyjnymi cechami fizycznymi. 

Wymiarem technicznym weryfikującym dojrzałość tych ustaleń było obudowanie ich pełną architekturą *MLOps*, opartą na konteneryzacji Docker i automatycznych audytach wycieku danych (Pytest). Zastosowanie środowisk `jupyter` do R&D oraz wyizolowanych potoków uruchomieniowych `pipeline-full` zapewniło absolutną reprodukowalność całego procesu badawczego. Co więcej, zbudowanie interfejsu Streamlit zademonstrowało, w jaki sposób sztuczna inteligencja może wejść do codziennej pracy personelu nietechnicznego poprzez interaktywne raportowanie sił SHAP. 

Zgodnie z dobrymi praktykami oprogramowania naukowego, cały kod źródłowy projektu, wraz z konfiguracją Docker Compose oraz testami jednostkowymi, został przygotowany w sposób otwarty. Gwarantuje to możliwość bezproblemowego zweryfikowania, zreplikowania lub rozwinięcia badań w przyszłości.

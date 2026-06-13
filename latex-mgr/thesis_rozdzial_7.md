# 7. Aplikacja demonstracyjna i sposób prezentacji wyników

Zbudowanie wysoce skutecznego algorytmu analitycznego w środowisku konsolowym (opisanym w poprzednich rozdziałach) rozwiązuje problem z punktu widzenia *data science*, lecz nie dostarcza jeszcze użytecznego narzędzia dla pracowników schroniska. Aby zniwelować barierę wejścia i udowodnić praktyczną aplikowalność opracowanych modeli, stworzono interaktywną aplikację demonstracyjną (Dashboard). 

W niniejszym rozdziale zaprezentowano cele, architekturę oraz przykładowe scenariusze użycia tego narzędzia wizualnego, które stanowi zwieńczenie praktycznej części pracy.

## 7.1. Cel i środowisko aplikacji

Głównym celem aplikacji jest umożliwienie personelowi Austin Animal Center (AAC) natychmiastowego korzystania z dobrodziejstw predykcji maszynowej bez konieczności posiadania jakiejkolwiek wiedzy programistycznej czy analitycznej. 

Kluczowym założeniem projektowym, zgodnym z opisaną w Rozdziale 6 architekturą *Artifact-First*, jest pełna pasywność aplikacji w kontekście uczenia maszynowego. Aplikacja nigdy nie trenuje modeli interaktywnie. Zamiast tego zaciąga ona uprzednio zatwierdzone i zamrożone weryfikacją testową artefakty ze współdzielonego wolumenu Dockera `/models`.

Poniższy fragment kodu udowadnia lekkość warstwy konsumpcyjnej – interfejs potrzebuje zaledwie ułamka sekundy na wczytanie pre-trenowanego modelu i wejście w stan gotowości operacyjnej:

```python
import streamlit as st
import joblib

@st.cache_resource
def load_model():
    # Model ładuje się tylko raz po starcie kontenera "dashboard"
    return joblib.load('/app/models/adoption_model.joblib')

model = load_model()
```
*[Listing 7.1: Mechanizm buforowania wczytywania modelu zapobiegający degradacji wydajności]*

Dzięki temu system, działający jako izolowany kontener `dashboard`, jest całkowicie odporny na uszkodzenia wywołane wprowadzaniem błędnych danych przez operatora, a czas oczekiwania na prognozę dla schroniska jest natychmiastowy.

## 7.2. Główne moduły funkcjonalne (Widoki)

Interfejs użytkownika podzielono na logiczne sekcje odpowiadające najczęstszym operacjom w schronisku.

### 7.2.1. Single Animal Predictor (Symulator Pojedynczego Przyjęcia)

Jest to główny moduł operacyjny przeznaczony dla personelu pracującego przy tzw. "bramie" (intake). Umożliwia on manualne wprowadzenie cech zwierzęcia w momencie jego przyjmowania do ośrodka. 
Użytkownik definiuje z rozwijanych list m.in.:
* Gatunek, płeć oraz wiek (w miesiącach lub latach).
* Typ i okoliczności przyjęcia (np. zrzeczenie właścicielskie, znaleziony na ulicy, uratowany z wypadku).
* Cechy wyglądu (rasę dominującą oraz umaszczenie).

Po zatwierdzeniu formularza, wbudowany w aplikację skrypt mapuje wartości z interfejsu na odpowiedni, stabelaryzowany wektor wejściowy i przepuszcza go przez wiodący model klasyfikacyjny. Pracownik natychmiast otrzymuje jasny werdykt:
1. **Ocena ryzyka długiego pobytu:** Aplikacja klasyfikuje przypadek, używając rygorystycznie ustalonego progu decyzyjnego.
2. **Kwantyfikacja:** Prezentowane jest bazowe prawdopodobieństwo adopcji w procentach.

![Zrzut ekranu: Interfejs "Single Animal Predictor"](file:///C:/Users/paula/Documents/mgr pjatk/assets/streamlit_screenshot.png)
> [!NOTE]
> Ze względu na dynamiczny charakter aplikacji Streamlit, zrzut ekranu musi zostać wygenerowany manualnie podczas działania systemu i podmieniony pod ścieżką `assets/streamlit_screenshot.png` przed ostatecznym generowaniem pliku PDF.

### 7.2.2. Wyjaśnialność Decyzji (Interactive SHAP)

Wektor prawdopodobieństwa jest cenną informacją, jednak personel weterynaryjny wymaga zrozumienia motywów stojących za oceną algorytmu (tzw. problem *explainability*). Dlatego dla każdej wygenerowanej predykcji pojedynczego zwierzęcia aplikacja rysuje wykres sił SHAP (tzw. SHAP force plot lub waterfall plot). 

W perspektywie globalnej (całego zbioru), siły te układają się w logiczne wzorce, co prezentuje poniższy wykres:

![Wykres sił SHAP - podsumowanie klasyfikacji](file:///C:/Users/paula/Documents/mgr pjatk/reports/figures/shap_summary_classification.png)
*[Rysunek 7.1. Globalne podsumowanie wpływu najważniejszych cech (SHAP Summary) na decyzję klasyfikacyjną modelu. Kolor czerwony oznacza wyższą wartość danej cechy.]*

Pracownik na ekranie aplikacji widzi analogiczny, lecz wyizolowany wykres dla jednego zwierzęcia. Widzi na nim, które konkretnie atrybuty (np. podeszły wiek psa, ciemne umaszczenie, czy specyficzna trudna sytuacja znalezienia) pociągnęły wynik w dół, a które (np. bycie w typie rasy pożądanej) zadziałały na jego korzyść. Jest to fundament zaufania między człowiekiem a algorytmem, zapobiegający odrzuceniu systemu zjawiskiem potocznie zwanym "AI black-box fatigue".

### 7.2.3. Eksplorator Kohort Historycznych

Dla stanowisk kierowniczych aplikacja udostępnia widok umożliwiający przeglądanie historycznych podgrup (kohort) populacji AAC. Pozwala to na nałożenie filtrów i obserwację, jak zmieniały się krzywe adopcyjności oraz mediany długości pobytu w czasie. Aplikacja potrafi zidentyfikować "najsłabsze ogniwa" systemu (np. nagromadzenie się starszych kotów w określonym kwartale roku) bez konieczności generowania zapytań SQL przez analityka.

## 7.3. Scenariusze Użycia w Pracy Schroniska (Triaging)

Największą wartością dodaną aplikacji z punktu widzenia operacyjnego nie jest precyzyjna predykcja kalendarzowa, lecz zastosowanie zasady selekcji priorytetowej (tzw. triaging). 

W tradycyjnym środowisku każdy przyjmowany pies otrzymuje standardową opiekę. Czasem mija kilka miesięcy, zanim personel zauważy, że dane zwierzę popadło w depresję schroniskową, stając się "rezydentem". W scenariuszu wspartym aplikacją:
1. Zwierzę jest wprowadzane do bazy.
2. Aplikacja natychmiast flaguje je jako przypadek "skrajnie trudny" (bardzo niskie prawdopodobieństwo adopcji, długi estymowany czas pobytu).
3. Pracownik już pierwszego dnia widzi ten alert i uruchamia procedury zaradcze: np. przekazanie zwierzęcia do tymczasowego domu zastępczego (foster), rozpoczęcie wzmożonej promocji marketingowej w lokalnych mediach społecznościowych czy szybką interwencję behawiorysty.

Ograniczenia aplikacji pozostają tożsame z ograniczeniami samego modelu – jeżeli profil zwierzęcia wykracza poza ramy historyczne na których uczono system (np. całkowicie nowa rasa egzotyczna w regionie), narzędzie wygeneruje błąd lub poprosi o przyporządkowanie zwierzęcia do szerszej grupy bazowej. Aplikacja unika maskowania niepewności – w przypadku braku wystarczających danych woli wyświetlić ostrzeżenie, niż wprowadzać personel w błąd "pewną" predykcją opartą na zgadywaniu.

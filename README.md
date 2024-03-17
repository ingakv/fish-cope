# Prosjektbeskrivelse

Videoovervåkning av vassdrag er viktig for forvaltning. Stiftelsen Anadrom
(www.anadrom) har i samarbeid med Nordavind Utvikling i Troms utviklet en
pakkeløsning for videoovervåking av vassdrag. Denne løsningen tar video (bilder) som
brukes for å få oversikt over antall fisk, størrelse på fisken og arter. Systemet i dag er
basert på en sensor, når fisken passerer sensoren begynner videoopptaket, og
videoen lagres før den lastes opp til en server.

Siden sensoren som trigger opptak er basert på «blokkering», så vil andre ting enn fisk
også starte et opptak, slik som løv, kvister, etc. Det er ønskelig å lage et system som
er mer robust, det vil si at man kun tar opptak av fisk. Dette kan gjøres ved at man
analyserer videoen (bildene) og hvis man ser en fisk så skal man starte opptaket.
Ønsket er å gå bort fra å bruke sensorer, og kun basere det på video/bilder. En slik
løsning, hvis den er robust og presist, vil gjøre etterarbeidet med videoene effekti vt,
raskere og mer presist. Dette vil i tillegg gjøre videoovervåkningsløsningen billigere å
produsere.

Det som må gjøres i prosjektet er å detektere fisk/ikke fisk i videostrømmen. Dette må
skje raskt, slik at man kan starte et opptak. I tillegg må dette være en løsning som kan
kjøre på lav datakraft (for eksempel på en Raspberry Pi). Robusthet med høy presisjon
er viktig. Løsningen må fungere under forskjellige lysforhold, forskjellig sikt i vannet,
forskjellige vannstand (turbulens i vannet), etc. Anadrom stiller med datamateriale
(video) for oppgaven.

Mål: Deteksjon av fisk/ikke fisk i video med høy presisjon under forskjellige forhold.

Tilleggsoppgaver kan være:

- Telle antall fisk i videoen
- Finne hvilken vei fisken svømmer (oppstrøms eller nedstrøms)
- Mulighet for å filtrere bort mindre fisk (for eksempel ikke starte opptak ved
    ungfisk/smolt)
- Detektere andre dyr, slik som mink, oter, fugl, i klippene
- Artsdeteksjon

Stiftelsen Anadrom har i samarbeid med Nordavind Utvikling i Troms utarbeidet en
teknisk beskrivelse av videoovervåkingssystemet med oppsummering av erfaringer og
utfordringer samt en ønsket videreutvikling i faser.

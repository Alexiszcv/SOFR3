# Brief d'implémentation — Régression du spread SOFR–IORB sur les flux du Trésor

## Objectif

Construire un panel **journalier** à partir des fichiers de `Data/`, puis estimer une régression
linéaire qui explique/forecast la **variation quotidienne du spread SOFR–IORB** à partir du drain de
réserves du Trésor et de l'état du système de réserves. Sortie attendue : un `DataFrame` journalier
propre + un modèle OLS estimé avec diagnostics.

## Spécification cible

Cible (en **différence**, pas en niveau) :

    y_t = Δ(SOFR − IORB)_t

Régression :

    y_t = α + β1·tga_drain_t
            + β2·(tga_drain_t × reserves_centered_t)
            + β3·gross_settlement_t
            + β4·rrp_t
            + γ'·D_t
            + ε_t

avec D_t = dummies {fin de mois, fin de trimestre, dates de taxes}, et les jours **FOMC exclus**.

Priors de signe (à vérifier) : β1 > 0 (drainer les réserves pousse SOFR au-dessus de l'IORB) ;
β2 < 0 (plus les réserves sont abondantes, plus la sensibilité au drain est faible — convexité de la
courbe de demande de réserves) ; β3 > 0 (plus de bilan dealer mobilisé → pression repo) ;
β4 < 0 (RRP élevé = tampon de liquidité).

## Données (toutes dans `Data/`)

| Fichier | Source | Extraire | Unité | Fréquence |
|---|---|---|---|---|
| `SOFR.csv` | FRED | `SOFR` | % | quotidien |
| `IORB.csv` | FRED | `IORB` | % | quotidien (depuis 2021-07-29) |
| `RRPONTSYD.csv` | FRED | take-up RRP | millions $ | quotidien |
| `WRESBAL.csv` | FRED | réserves | millions $ | **hebdomadaire** |
| `DTS_OpCashBal_*.csv` | DTS | closing balance TGA → `tga_drain` | millions $ | quotidien |
| `DTS_PubDebtTrans_*.csv` | DTS | issues / redemptions marketable → `gross_settlement` | millions $ | quotidien |
| `DTS_OpCashDpstWdrl_*.csv` | DTS | (option) recettes fiscales | millions $ | quotidien |
| `treasury.csv` | TreasuryDirect | (option) `dealer_take_$` par date de settlement | $ | par enchère |

## Étapes

### 1. Charger et nettoyer chaque source en une série journalière indexée par date

- **FRED (SOFR, IORB, RRP, WRESBAL)** : `Record Date`/`DATE` → datetime ; valeur → numérique
  (`errors='coerce'`, certains jours fériés contiennent un `.`). RRP et WRESBAL : diviser par 1000
  pour passer en $bn. WRESBAL est **hebdomadaire** → reindexer sur le calendrier journalier et
  **forward-fill** (c'est une variable d'état lente, c'est correct de la propager).
- **DTS — Operating Cash Balance** : nettoyer le BOM des entêtes ; filtrer
  `Type of Account` contenant `"Closing Balance"` ; la valeur du jour est dans la colonne
  (mal nommée) `Opening Balance Today` ; /1000 → $bn ; trier par date ; `tga_drain = close.diff()`
  (signe : + = réserves drainées).
- **DTS — Public Debt Transactions** : filtrer `Security Marketability == "Marketable"` ET
  `Security Type ∈ {Bills, Notes, Bonds}` (exclut Government Account Series = intragouvernemental,
  FFB, et l'increment TIPS qui est un accrual, pas du cash) ; pivoter `Transaction Type` en colonnes
  Issues / Redemptions sur `Transactions Today` (/1000 → $bn) ;
  `gross_settlement = issues + redemptions`. (Calculer aussi `net_issuance = issues − redemptions`
  mais **seulement comme variable descriptive** — voir Pièges.)
- **DTS — Deposits/Withdrawals (option)** : `tax_receipts` = somme des catégories de dépôt dont le
  libellé matche taxes (les libellés ont changé dans le temps : `Taxes - *`, `Cash FTD's Received`,
  `Individual Income and Employment Taxes, Not Withheld`, etc.).
- **treasury.csv (option)** : auto-détecter le délimiteur (tab / virgule / point-virgule = le plus
  fréquent dans l'entête) ; si non-virgule, remplacer la virgule décimale par un point avant
  conversion numérique ; `dealer_take_$` = somme de `PrimaryDealerAccepted` groupée par `IssueDate`
  (date de settlement, **pas** la date d'enchère).

### 2. Construire la cible

- `spread = SOFR − IORB`
- `y = spread.diff()`
- Note : sur les jours non-FOMC, `Δspread ≡ ΔSOFR` (l'IORB est constant entre réunions). La cible
  reste le spread, pas ΔSOFR seul, pour rester robuste quand les taux bougent.

### 3. Calendrier et merge

- Épine dorsale = calendrier des jours ouvrés (ou les dates de `SOFR`).
- Left-join de toutes les séries sur `date`.
- Forward-fill WRESBAL après le merge (hebdo → journalier).
- Gérer les NaN de bord (début de série IORB en 2021-07 ; premières diffs).

### 4. Dummies et régime

- **FOMC** : coder en dur les dates de réunion 2021–2026 → dummy, et **exclure** ces jours de
  l'estimation (l'IORB y saute, ΔSOFR y est dominé par la politique, pas par les flux).
- **Fin de mois / fin de trimestre** : dummies (window dressing, contraintes de bilan dealer).
- **Dates de taxes** : dummies (mi-mars/avril/juin/sept + échéances trimestrielles) — ou s'appuyer
  sur les pics empiriques de `tax_receipts` si l'option a été construite.

### 5. Régresseurs et interaction

- Centrer (ou standardiser) `reserves` → `reserves_centered`, pour que β1 s'interprète au niveau
  moyen de réserves.
- `interaction = tga_drain × reserves_centered` : capture que la **sensibilité** du spread au drain
  dépend du régime de réserves (le plus gros gain de R² attendu).
- Ne PAS inclure `net_issuance` en plus de `tga_drain` (redondant — c'est une sous-composante).

### 6. Estimer

- OLS (`statsmodels`), erreurs-types **HAC / Newey-West** (autocorrélation des taux journaliers).
- `y ~ tga_drain + interaction + gross_settlement + rrp + dummies`, jours FOMC exclus.
- Option : ajouter `dealer_take_$` (canal bilan dealer, secondaire).

### 7. Diagnostics

- R², R² ajusté ; significativité et **signe** des coefficients vs priors.
- **VIF** (multicolinéarité) : surveiller `tga_drain` vs `interaction`, et `tga_drain` vs
  `gross_settlement`.
- Autocorrélation des résidus (Durbin-Watson / Ljung-Box), hétéroscédasticité.
- Graphes : fitted vs actual ; résidus dans le temps (repérer les pics fin-de-trimestre non captés).
- **Out-of-sample** : split train/test chronologique (ou forecast roulant), RMSE — c'est le vrai
  juge puisque l'objectif est le forecast.

### 8. Robustesse / itérations

- Remplacer l'interaction par un **split de régime** (réserves hautes vs basses) et comparer.
- Avec / sans jours FOMC ; avec / sans dummies.
- Comparer `tga_drain` vs `net_issuance` vs `gross_settlement` comme régresseur principal pour voir
  lequel resserre le nuage.

## Pièges (consolidés)

- **DTS en millions** → tout /1000 pour homogénéiser en $bn ; entêtes avec BOM à nettoyer ; colonne
  valeur `Opening Balance Today` mal nommée ; filtrer Marketable et exclure l'intragouvernemental.
- **WRESBAL hebdomadaire** → forward-fill ; c'est une variable d'état lente (régime), pas un flux.
- **IORB constant entre FOMC** → cible en **différence** ; **exclure les jours FOMC** (sinon outliers
  à fort levier qui faussent la pente).
- **treasury.csv** : délimiteur à auto-détecter + virgule décimale possible (export FR).
- **Redondance** : jamais `tga_drain` ET `net_issuance` ensemble. `tga_drain` (canal réserves) +
  `gross_settlement` (canal bilan dealer) sont, eux, complémentaires.
- **Newey-West** obligatoire (résidus autocorrélés).
- **Hors périmètre** : BTC z-score et tail/stop-through n'entrent PAS dans cette régression (ils
  appartiennent au projet event-study enchère → rendement → FX). `treasury.csv` ne sert ici qu'au
  régresseur optionnel `dealer_take_$`.
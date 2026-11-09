# Publier PI-Loc — mode d'emploi (≈ 30 min)

Tout est prêt. Il reste à faire, depuis ton compte, ce que je ne peux pas faire à ta place.

## 1. GitHub (10 min)

1. Sur github.com, crée un dépôt **public** vide nommé `pi-loc`, sans README.
2. Dans un terminal, dans le dossier `pi-loc` (décompressé depuis `pi-loc-github.zip`) :
   ```bash
   git remote add origin https://github.com/yulliwasameur/pi-loc.git
   git branch -M main
   git push -u origin main
   ```
3. Le fichier `data/piloc.jsonl` n'est **pas** poussé (voir `DATA_CARD.md` : deux sources sans licence déclarée). N'importe qui le reconstruit avec `python -m piloc.build_data`.

## 2. Zenodo (10 min)

1. Sur zenodo.org, connecte-toi avec GitHub, puis **Settings → GitHub** : active le dépôt `pi-loc`.
2. Sur GitHub, crée une **release** `v1.0.0`. Zenodo l'archive tout seul et donne un DOI.
3. Le fichier `.zenodo.json` remplit déjà le titre, l'auteur, l'ORCID, les mots-clés et la licence.

## 3. Mettre l'URL et le DOI dans le papier (5 min)

Dans `paper/main.tex`, section *Data and Code Availability*, remplace les deux marques rouges :
- `\TODO{GitHub URL}` → `\url{https://github.com/yulliwasameur/pi-loc}`
- `\TODO{Zenodo DOI}` → `\url{https://doi.org/10.5281/zenodo.XXXXXXX}`

Puis recompile. La lettre d'accompagnement a les mêmes deux trous.

## 4. HAL puis IEEE Access

1. Dépose le PDF sur HAL (affiliation Efrei, Cnam en second).
2. Soumets sur IEEE Access (ScholarOne) avec le PDF, les sources LaTeX et la lettre.

## Avant de soumettre : à faire par toi

- **Relire le code et les résultats** (au moins `piloc/audit.py` et `results/summary.json`). La phrase des remerciements dit que tu l'as fait.
- Décider si tu publies aussi `data/piloc.jsonl` sur Zenodo (voir `DATA_CARD.md`).

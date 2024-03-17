## Objectifs
BUT : suivre l'évolution à partir d'une photo et/ou d'une vidéo de la tension d'un fluide savonneux.
### Fonctions annexes :
- traitement d'image : décomposition de l'image et identification des coins.
  - Resources : 
    - [Imageio](https://imageio.readthedocs.io/en/stable/)
    - [Pillow](https://pillow.readthedocs.io/en/stable/)
### Rendu : 
On veut l'évolution de l'épaisseur, donc à une image, on renvoie la même avec en noir les 
points d'épaisseur maximale (relative) et sur un gradient de gris les points du fluide, 
jusqu'à ceux d'épaisseur nulle en blanc.
## Process:
⚠️ Ne pas enlever le noir : 
la frange noire permet de déterminer l'épaisseur zéro, donc l'épaisseur absolue du film.
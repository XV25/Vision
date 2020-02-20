# Notes sur l'article baker 11


## Principe
Méthode de flots optiques consiste en **l'optimisation de la fonciton d'énergie**.

 Equation d'énergie : 


$$ E_{Global} = E_{Data} - \lambda E_{Prior} $$

### Data term

#### Brithness constancy
D'une image  l'autre, l'intensité ou la couleur ne change pas.

$$ I (x, y, t) = I (x +u,  y+v, t+1)$$

Linéarisation :

$$ I (x, y, t) = I (x, y, t) + u\frac{\partial I}{\partial x} + v\frac{\partial I}{\partial y} + 1\frac{\partial I}{\partial t}$$

D'où la constante de flot optique :

$$ u\frac{\partial I}{\partial x} + v\frac{\partial I}{\partial y} + 1\frac{\partial I}{\partial t}=0$$


## Idées:
Flot optic but : déterminer le mouvement des objest dans une image

Utilisation : stabilisation optique, dplacement drone, 3D (IMAGE PROCHE CAM BOUGE PLUS VITE QUE arriere plan, segmentation, frame interpolation (pour fiare des raletit ou rajouter des frames)

Hypothèses : la luminosité ne change pas trop d'une image à l'autre,
petit nombre de frame

texture uniforme (balle range qui tourne : impossible à voir le mouvement car ne change pas 

## Calculation :
Horn and Shunk : look only in the local neighbohood pour estimer en "moyennant les déplacements des pixels alentours(smoothing effect)  donc c'est une approche globale du mouvzment de l'image, on ne s'intereese pas à chaque pixels en particulier. approach pour déterminer uv (coordonnées vitesse) pour les petits deplacement (moins de 10 pixels entre 2 images) de l'ordre du pixel;
Permet de traiter des grandes images ou des endroit don on ne pourrait pas de manière evidente trouver le mouvement (cf balle orange uniforme) 

Lucas Kanade : sur chaque pixel autour du pixel que l'on regardes

aperture problem

Solution si image bouge trop vite : reduire sa taille (fait des moyennes locales des pixels => ca les raproches
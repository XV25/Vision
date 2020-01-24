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

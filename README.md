# Vision

Résumé démarche du projet : 

- Prendre une vidéo en temps réel.
- Sur première frame de vidéo, repérer objets d'intérêt. Récupérer dans une fenêtre les pixels associés à cet objet
- Sur la frame suivante, appliquer HS : en déduire fenêtre encadrant l'objet, et vecteur déplacement.
- Continuer la démarche sur toutes les frames suivants.

# Liens utiles

* Pour la gestion de la vidéo avec la webcam : [cliquer ici](https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html)
* Méthode Lucas Kanade (bout de code py et exemples) : https://sandipanweb.wordpress.com/2018/02/25/implementing-lucas-kanade-optical-flow-algorithm-in-python/

Rappel tâches : 

- Fonctionnement webcam --> Nathan
- Détection couleur dans zone --> Erwann 
- Implémentation LK / HS --> Coraline

"""structrisk : bibliothèque de pricing d'options et de produits structurés.

Ce package regroupe les briques utilisées en gestion des risques de marché :
- pricing Black-Scholes-Merton et Grecques (ordre 1 et ordre supérieur) ;
- vol implicite et surface de volatilité ;
- arbre binomial (CRR) européen / américain ;
- moteur Monte Carlo vectorisé multi-actifs ;
- payoffs de produits structurés (autocall, reverse convertible, bonus cap,
  note à capital protégé, worst-of) ;
- valorisation Monte Carlo des structures et Grecques associées ;
- agrégation des Grecques d'un portefeuille ;
- simulation de couverture en delta discrète.

Tout le code est déterministe (seed explicite), vectorisé numpy et fonctionne
hors ligne : aucune donnée de marché n'est téléchargée, tous les paramètres
(spot, taux, dividendes, vol) sont synthétiques mais réalistes et définis
dans les scripts d'exemple.
"""

__version__ = "1.0.0"

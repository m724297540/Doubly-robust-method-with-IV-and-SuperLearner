# Doubly-robust-method-with-IV-and-SuperLearner
This method is an extension of the machine learning based doubly robust IV approach proposed by *Syrgkanis, Vasilis, et al. "Machine learning estimation of heterogeneous treatment effects with instruments." Advances in Neural Information Processing Systems 32 (2019)*.

This method incorporates the **SuperLearner** R package in the first stage to obtain nuisance parameters and preliminary estimates. Instead of manually choosing a candidate estimator for the first stage, this method uses K-fold cross-validation to select a functional form of candidate estimators from a set of candidat estimators.

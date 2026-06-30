Features:

a) Add new training loss (stop gradient only for predictors, toggleable) - Done
b) Add new eval decomposition at the best validation point of the training (and after 10% of the training)
    i) Compute the truly optimal weights - Done
    ii) Train N delegator ensembles (same architecture, same data split) - Almost done
    iii) Pick the best delegator ensemble, calculate the regret etc - Done 
c) Keep track only of individual errors and diversity:
    i) In predictors: weighted individual errors and weighted diversity and soft number of selected models
    ii) In delegators: FROM THE optimal weights given by the RESTRICTED ORACLE -> individual errors and diversity (uniform weights)


Todo:

1) Fix the regression training inequations in text right now they assume scalar y, implement norm
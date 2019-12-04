
## Get the overall AUC and PRAUC score for the seperate models.
## Doing 10-fold cross validation:
def combinedROC():

    x = x_compressed.repeat(20, axis = 0)
    y_combined = ys['combined'].repeat(20)
    ym =  ys['Medium'].repeat(20)
    ye =  ys['Evrn'].repeat(20)
    cv = StratifiedKFold(n_splits = 10)
    splits = cv.split(x, y_combined)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    plt.figure()
    k = 0
    for train, test in splits:
        k+=1
        x_tr = x[train]
        x_te = x[test]
        ym_tr = ym[train]
        ym_te = ym[test]
        ye_tr = ye[train]
        ye_te = ye[test]
    
        sample_size = (x_te.shape)[0]

        ## Fit the model for 'Medium'
        mmodel = svm.SVC(gamma = 'auto', random_state = 256, probability=True)
        mmodel.fit(x_tr, ym_tr)
        mproba = mmodel.predict_proba(x_te)
        n_mproba = (mproba.shape)[1]            # Number of all the unique class in Medium

        ## Fit the model for 'Evrn'
        emodel = svm.SVC(gamma = 'auto', random_state = 256, probability=True)
        emodel.fit(x_tr, ye_tr)
        eproba = emodel.predict_proba(x_te)
        n_eproba = (eproba.shape)[1]            # Number of all the unique classes in Stress

        ## Combined two predictions together
        proba_combined = np.zeros((sample_size, n_mproba*n_eproba))

        ## Get all possible combination of two different labels:
        ls1 = np.unique(ys['Medium'])
        ls2 = np.unique(ys['Evrn'])
        classes = [i + '_' + j for i in ls1 for j in ls2]

        ## Calculate the joint probability by multiply two numpy array
        ## The new array has the shape of: n_sampl * (number of all the possible combinations for the two labels)
        for i in range(n_mproba):
            proba_combined[:, i*n_eproba:(i+1)*n_eproba] = (mproba[:, i]).reshape(-1, 1) * eproba

        ## Binarize the combined value
        test_bi = label_binarize(y_combined[test], classes = classes).ravel()

        ## Calculate roc by macro average
        fpr, tpr, _ = roc_curve(test_bi, proba_combined.ravel())
        auc_ = auc(fpr, tpr)
        aucs.append(auc_)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        
        plt.plot(fpr, tpr, label = 'Fold: %d, AUC: %.2f'%{k, auc_})

    plt.legend()
    plt.show()
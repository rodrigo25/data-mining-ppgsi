function [] = main_spambase_kfold_adaboost()

    fileName = pre_processing('data_spambase',0);
    load(fileName);
    
    K = preProcInfo.k;
    classes = preProcInfo.classes;
    
    T = 5; % adaboost rounds
    h = 10; % adaboost MLP components hidden layer neuron count
    maxEpochs = 500; % adaboost MLP components max epoch count
    
    Yh = adaboostM2(Xtr, Ytr, Xval, Yval, Xtest, Ytest, classes, T, h, maxEpochs, 0, 0, 0);
   
    fprintf('Adaboost global answer (%d components, %d hidden layer neurons, %d epochs)\n', T, h, maxEpochs);
    [~,Yh]= max(Yh,[],2);
    [~,Ytest] = max(Ytest,[],2);
    multiclassConfusionMatrix( Ytest, Yh, classes, 1, 'Adaboost' );
    
    % MLP normal para comparar
    [ A, B ] = MLPtreina( Xtr, Ytr, Xval, Yval, h, maxEpochs, .01, 0, 1);
    Y = MLPsaida( Xtest, A, B );
    [~,Y]= max(Y,[],2);

    fprintf('MLP answer (%d hidden layer neurons, %d epochs)\n', h, maxEpochs);
    multiclassConfusionMatrix( Ytest, Y, classes, 2, 'MLP' );
end
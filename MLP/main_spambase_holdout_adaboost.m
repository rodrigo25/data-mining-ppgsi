function [] = main_spambase_holdout_adaboost()

    T = 5; % adaboost rounds
    h = 5; % adaboost MLP components hidden layer neuron count
    nepocas = 50; % adaboost MLP components max epoch count
    
    %fileName = pre_processing('data_spambase',1);
    %load(fileName);
    load('data/data_spambase.mat');
    % normalizes data
    
    [X, ~, ~] = normalization( X, 'zscore' );
    %[X, ~, ~, min_val, max_val] = normalization( X, 'minmax' );
    
    [Y, classes] = multiclassY(Y);
    
    [ Xtr, Ytr, Xtest, Ytest ] = holdout( X, Y, 0.7 );
    [ Xtr, Ytr, Xval, Yval ] = holdout( Xtr, Ytr, 0.99 );

    Yh = adaboostM2(Xtr, Ytr, Xval, Yval, Xtest, Ytest, classes, T, h, nepocas, 0, 0, 0);
   
    fprintf('Adaboost global answer (%d components, %d hidden layer neurons, %d epochs)\n', T, h, nepocas);
    [~,Yh]= max(Yh,[],2);
    [~,Ytest] = max(Ytest,[],2);
    multiclassConfusionMatrix( Ytest, Yh, classes, 1, 'Adaboost' );
    
    % MLP normal para comparar
    [ A, B ] = MLPtreina( Xtr, Ytr, Xval, Yval, h, nepocas, .01, 0, 1);
    Y = MLPsaida( Xtest, A, B );
    [~,Y]= max(Y,[],2);

    fprintf('MLP answer (%d hidden layer neurons, %d epochs)\n', h, nepocas);
    multiclassConfusionMatrix( Ytest, Y, classes, 2, 'MLP' );
end
function [] = main_spambase_holdout_adaboost()

    T = 5; % adaboost rounds
    h = 5; % adaboost MLP components hidden layer neuron count
    nepocas = 10; % adaboost MLP components max epoch count
    
    load('data/data_spambase.mat')

    % normalizes data
    
    [X, ~, ~] = normalization( X, 'zscore' );
    %[X, ~, ~, min_val, max_val] = normalization( X, 'minmax' );
    
    [Y, classes] = multiclassY(Y);
    
    [ Xtr, Ytr, Xtest, Ytest ] = holdout( X, Y, 0.7 );

    Yh = adaboostM2(Xtr, Ytr, Xtest, Ytest, classes, T, h, nepocas);
   
    fprintf('Adaboost global answer (%d components, %d hidden layer neurons, %d epochs)\n', T, h, nepocas);
    [~,Yh]= max(Yh,[],2);
    [~,Ytest] = max(Ytest,[],2);
    multiclassConfusionMatrix( Ytest, Yh, classes, 'Adaboost' );
    
    % MLP normal para comparar
    [ A, B ] = MLPtreina( Xtr, Ytr, [], [], h, nepocas);
    Y = MLPsaida( Xtest, A, B );
    [~,Y]= max(Y,[],2);

    fprintf('MLP answer (%d hidden layer neurons, %d epochs)\n', h, nepocas);
    multiclassConfusionMatrix( Ytest, Y, classes, 'MLP' );
end
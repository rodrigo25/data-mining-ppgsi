function [] = main_winequality_holdout_adaboost()

    T = 20; % adaboost rounds
    h = 30; % adaboost MLP components hidden layer neuron count
    nepocas = 1000; % adaboost MLP components max epoch count
    
    load('data_winequality-red.mat');

    % normalizes data
    
    [X, ~, ~] = normalization( X, 'zscore' );
    %[X, ~, ~, min_val, max_val] = normalization( X, 'minmax' );
    
    [Y, classes] = multiclassY(Y);
    
    holdoutFileName = 'winequality_red_holdout.mat';
    if exist(holdoutFileName,'file') == 2
        load(holdoutFileName);
    else
        [ Xtr, Ytr, Xtest, Ytest ] = holdout( X, Y, 0.7 );
        save('winequality_red_holdout', 'Xtr', 'Ytr', 'Xtest', 'Ytest'); 
    end

    Yh = adaboostM2(Xtr, Ytr, Xtest, Ytest, classes, T, h, nepocas, 0, 1, 0);

    fprintf('Adaboost global answer (%d components, %d hidden layer neurons, %d epochs)\n', T, h, nepocas);
    [~,Yh]= max(Yh,[],2);
    [~,Ytest] = max(Ytest,[],2);

    multiclassConfusionMatrix( Ytest, Yh, classes, 2, 'Adaboost' );
    
    % MLP normal para comparar
    [ A, B ] = MLPtreina( Xtr, Ytr, [], [], h, nepocas);
    Y = MLPsaida( Xtest, A, B );
    [~,Y] = max(Y,[],2);
    fprintf('MLP answer (%d hidden layer neurons, %d epochs)\n', h, nepocas);
    
    multiclassConfusionMatrix( Ytest, Y, classes, 3, 'MLP' );
end
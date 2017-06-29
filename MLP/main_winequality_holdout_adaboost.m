function [] = main_winequality_holdout_adaboost()

    T = 100; % adaboost rounds
    h = 10; % adaboost MLP components hidden layer neuron count
    nepocas = 1000; % adaboost MLP components max epoch count
    
    % preprocessa data_winequality-red
    % 0 = nao sobrescreve se existir
    fileName = pre_processing('data_winequality-red',0,'holdout');
    load(fileName);
   
    %K = preProcInfo.k;
    classes = preProcInfo.classes;

    [Yh, ~] = adaboostM2(Xtr, Ytr, Xval, Yval, Xtest, Ytest, classes, T, h, nepocas, 1);

    fprintf('Adaboost global answer (%d components, %d hidden layer neurons, %d epochs)\n', T, h, nepocas);
    [~,Yh]= max(Yh,[],2);
    [~,Ytest] = max(Ytest,[],2);

    multiclassConfusionMatrix( Ytest, Yh, classes, 2, 'Adaboost' );
    
    % MLP normal para comparar
    [ A, B ] = MLPtreina( Xtr, Ytr, Xval, Yval, 1, h, nepocas, .01, 0, 1);
    %[A,B] = treina_rede(Xtr,Ytr,h,nepocas);
    %[ A, B, ~] = treinamento(Xtr,Ytr,h,nepocas);
    
    Y = MLPsaida( Xtest, 1, A, B );
    [~,Y] = max(Y,[],2);
    fprintf('MLP answer (%d hidden layer neurons, %d epochs)\n', h, nepocas);
    
    multiclassConfusionMatrix( Ytest, Y, classes, 3, 'MLP' );
    
    %{
    for t=1:T
       Xtrained = XYtrained{t,1}; 
       Ytrained = XYtrained{t,2};
       [ A, B ] = MLPtreina( Xtr, Ytr, Xval, Yval, 1, h, nepocas, .01, 0, 1);
       Y = MLPsaida( Xtrained, 1, A, B );
       [~,Y] = max(Y,[],2);
       [~,Ytrained] = max(Ytrained,[],2);
       fprintf('Comparacao MLP t=%d\n', t);
       multiclassConfusionMatrix( Ytrained, Y, classes );
    end
    %}
end
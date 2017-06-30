function [] = main_winequality_holdout_adaboost()

    % preprocessa data_winequality-red
    % 0 = nao sobrescreve se existir
    fileName = pre_processing('data_winequality-red',0,'holdout');
    load(fileName);

    h = 10; % MLP components hidden layer neuron count
    nepocas = 1000; % adaboost MLP components max epoch count
    classes = preProcInfo.classes;
    
    Xtr = [Xtr;Xval];
    Ytr = [Ytr;Yval];
    
    %PCA
    %{
    [ coefs, ~, pcvar ] = pca(Xtr);           
    relVar = pcvar / sum(pcvar);
               
    for dim=1:length(relVar)
        if sum(relVar(1:dim)) >= 0.95
            break;
        end
    end
    Xtr = Xtr * coefs(:,1:dim);
    Xtest = Xtest * coefs(:,1:dim);
    Xval = Xval * coefs(:,1:dim);
    %}
    
    %{
    accs = zeros(size(Xtr,2),1);
    for i=1:size(Xtr,2)
        [ A, B ] = MLPtreina( Xtr(:,i), Ytr, Xval(:,i), Yval, 1, h, nepocas, .01, 0, 0);    
        accs(i) = multiclassConfusionMatrix( YtestClasses, Yh, classes, 3, 'MLP' );
    end
    [~,accsIndex] = sort(accs);
    
    featureCount = 1;
    features = [accsIndex(1)]  
    for i=2:length(accsIndex)
        if ismember(i,features)
           continue; 
        end
    end       
    %}
    
    % MLP normal para comparar
    [ A, B ] = MLPtreina( Xtr, Ytr, Xval, Yval, 1, h, nepocas, .01, 0, 0);
    % [A,B] = MLP_clodoaldo(Xtr,Ytr,h,nepocas);
    %[ A, B] = MLP_alfaAdaptativo(Xtr,Ytr,Xval,Yval,h,nepocas,0);
    
    Yh = MLPsaida( Xtest, 1, A, B );
    [~,Yh] = max(Yh,[],2);
    fprintf('MLP answer (%d hidden layer neurons, %d epochs)\n', h, nepocas);
    [~,YtestClasses] = max(Ytest,[],2);
    mlpAcc = multiclassConfusionMatrix( YtestClasses, Yh, classes, 3, 'MLP' );
    fprintf('MLP accuracy %f\n', mlpAcc);
    %pause;
    T = 100; % adaboost rounds
     
    %K = preProcInfo.k;

    [Yh, ~] = adaboostM2(Xtr, Ytr, Xval, Yval, Xtest, Ytest, classes, T, h, nepocas, 1);

    fprintf('Adaboost global answer (%d components, %d hidden layer neurons, %d epochs)\n', T, h, nepocas);
    [~,Yh]= max(Yh,[],2);
    [~,Ytest] = max(Ytest,[],2);

    ensembleAcc = multiclassConfusionMatrix( Ytest, Yh, classes, 3, 'Adaboost' );
    fprintf('MLP accuracy %f\n', mlpAcc);
    fprintf('Ensemble accuracy %f\n', ensembleAcc);
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
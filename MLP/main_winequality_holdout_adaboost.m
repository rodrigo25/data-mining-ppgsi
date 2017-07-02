function [] = main_winequality_holdout_adaboost()

    % preprocessa data_winequality-red
    % 0 = nao sobrescreve se existir
    fileName = pre_processing('data_winequality-red',0,'holdout');
    load(fileName);

    h = 10; % MLP components hidden layer neuron count
    
    classes = preProcInfo.classes;
    
    [~,YtestClasses] = max(Ytest,[],2);
    
    
    %Xtr = [Xtr;Xval];
    %Ytr = [Ytr;Yval];
    
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
    valError = zeros(size(Xtr,2),1);
    for i=1:size(Xtr,2)
        [ ~, ~, ~, MSEval, epochsExecuted ] = MLPtreina( Xtr(:,i), Ytr, Xval(:,i), Yval, 1, h, nepocas, .01, 0, 0);    
        valError(i) = MSEval(epochsExecuted);
    end
    [~,valErrorIndex] = sort(valError);
    
    features = valErrorIndex(1);
    featuresError = valError(valErrorIndex(1));
    for i=2:length(valErrorIndex)
        newFeatures = [features valErrorIndex(i)];
        [ ~, ~, ~, MSEval, epochsExecuted ] = MLPtreina( Xtr(:,newFeatures), Ytr, Xval(:,newFeatures), Yval, 1, h, nepocas, .01, 0, 0);
        newValError = MSEval(epochsExecuted);
        if newValError < featuresError
           features = newFeatures; 
           featuresError = newValError;
        end
    end       
    
    Xtr = Xtr(:,features);
    Xval = Xval(:,features);
    Xtest = Xtest(:,features);
    %}
    %Xtr = [Xtr;Xval];
    %Ytr = [Ytr;Yval];
    
    % MLP normal para comparar
    [ A, B ] = MLPtreina( Xtr, Ytr, Xval, Yval, h, 1000, 2, 1, 1 );
    % [A,B] = MLP_clodoaldo(Xtr,Ytr,h,nepocas);
    %[ A, B] = MLP_alfaAdaptativo(Xtr,Ytr,Xval,Yval,h,1000,0);
    
    Yh = MLPsaida( Xtest, A, B );
    [~,Yh] = max(Yh,[],2);
    fprintf('MLP answer (%d hidden layer neurons, %d epochs)\n', h, 1000);
    
    mlpAcc = multiclassConfusionMatrix( YtestClasses, Yh, classes, 2, 'MLP' );
    fprintf('MLP accuracy %f\n', mlpAcc);
    %return;
    T = 100; % adaboost rounds
     
    %K = preProcInfo.k;

    [Yh, Aen, Ben, beta, MSEtrain, MSEval] = adaboostM2(Xtr, Ytr, Xval, Yval, Xtest, Ytest, classes, T, h, 2000, 1);

    fprintf('Adaboost global answer (%d components, %d hidden layer neurons, %d epochs)\n', T, h, 2000);
    [~,Yh]= max(Yh,[],2);

    ensembleAcc = multiclassConfusionMatrix( YtestClasses, Yh, classes, 3, 'Adaboost' );

    [~, sortedComponents] = sort(MSEval);
    
    bestComponents = sortedComponents(1);
    bestMSE = MSEval(sortedComponents(1));
    bestBetas = beta(sortedComponents(1));
    for t=2:T
        newComponents = [bestComponents sortedComponents(t)];
        newBetas = [bestBetas beta(sortedComponents(t))];
        %newBetas = (newBetas * sum(beta)) ./ sum(newBetas);
        
        Yh = 0;
        for i=1:length(newComponents)
            componentIndex = newComponents(i);
            Ac = Aen{componentIndex};
            Bc = Ben{componentIndex};
            betac = newBetas(i);
            Yh = Yh + log(1/betac)*MLPsaida(Xval, Ac, Bc);
        end
        newErr = Yh-Yval;
        newMse = sum(sum( newErr.^2 ))/size(Xval,1);
        
        if bestMSE > newMse
           bestMSE = newMse;
           bestComponents = newComponents;
           bestBetas = newBetas;
        end
    end
    
    Yh = 0;
    for i=1:length(bestComponents)
        componentIndex = bestComponents(i);
        Ac = Aen{componentIndex};
        Bc = Ben{componentIndex};
        betac = beta(componentIndex);
        Yh = Yh + log(1/betac)*MLPsaida(Xtest, Ac, Bc);
    end
    [~,Yh]= max(Yh,[],2);
    selectedEnsembleAcc = multiclassConfusionMatrix( YtestClasses, Yh, classes, 2, 'MLP' );
    
    fprintf('MLP accuracy %f\n', mlpAcc);
    fprintf('Ensemble accuracy %f\n', ensembleAcc);
    fprintf('Ensemble accuracy (selected components) %f\n', selectedEnsembleAcc);
    fprintf('Ensemble #components (selected components) %d\n', length(bestComponents));

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
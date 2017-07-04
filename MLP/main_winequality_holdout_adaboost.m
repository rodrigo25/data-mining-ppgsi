function [] = main_winequality_holdout_adaboost()

    neurons = [20 40 60];
    epochs = [200 500 1000 5000];
    T = 100;
    numExecutions = 10;
    holdoutCount = 5;
    
    fileNames = pre_processing('data_winequality-red',0, 'holdout', holdoutCount);
    baseResultFileDir = 'ensemble-T100-arch0\winequality\red\holdout\';
    for fileIndex=1:numel(fileNames)
        load(fileNames{1}); 
        % assigns variables to themselves, so threads in the pool can see
        % them (matlab very cool and nice details)
        Xtest = Xtest;
        Xtr = Xtr;
        Xval = Xval;
        Ytest = Ytest;
        Ytr = Ytr;
        Yval = Yval;
        classes = preProcInfo.classes;
        [~,YtestClasses] = max(Ytest,[],2);

        resultFileDir = [baseResultFileDir num2str(fileIndex) '\'];
        if ~exist(resultFileDir, 'dir')
            mkdir(resultFileDir);
        end
       
        for i=1:length(neurons)
            h = neurons(i);
            for j=1:length(epochs)
                maxEpochs = epochs(j);
                YhEnsembles = cell(numExecutions,1);
                parfor (t=1:numExecutions,4)
                    resultFileName = [resultFileDir sprintf('maxEpochs%d_h%d_t%d', maxEpochs, h, t)];
                    if exist([resultFileName '.mat'], 'file') == 2
                       continue; 
                    end
                    Yh = adaboostM2(Xtr, Ytr, Xval, Yval, Xtest, Ytest, classes, T, h, maxEpochs, 0);
                    [~,Yh] = max(Yh,[],2);
                    YhEnsembles{t} = Yh;
                    acc = multiclassConfusionMatrix( YtestClasses, Yh, classes );
                    fprintf('h = %d\tfile = %d\tt = %d\tacc = %f\tmaxEpochs = %d\n', h, fileIndex, t, acc, maxEpochs);
                end
                for t=1:numExecutions
                    resultFileName = [resultFileDir sprintf('maxEpochs%d_h%d_t%d', maxEpochs, h, t)];
                    if exist([resultFileName '.mat'], 'file') == 2
                       continue; 
                    end
                    Yh = YhEnsembles{t};
                    acc = multiclassConfusionMatrix( YtestClasses, Yh, classes, 1, 'Adaboost', [resultFileName '_confusion_matrix'] );
                    save(resultFileName, 'acc', 'Yh');
                end
            end
        end
    end

    %{
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
    %}
end
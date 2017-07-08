function [] = main_winequality()
    % executa varias parametrizacoes de uma MLP com gradiente descendente e
    % método de bisseção
    % dados intermediarios sao persistidos assim que possivel 
    % sao criados graficos ao final da execucao
    neurons = [3 5 7 10:10:100];
    epochs = [200 500 1000];
    numExecutions = 20;
    holdoutCount = 5;
    saveData = 1;

    fileNames = pre_processing('data_winequality-red',0, 'holdout', holdoutCount);
    baseResultFileDir = 'grid\winequality\red\holdout\bisect\';
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
        
        resultFileDir = [baseResultFileDir num2str(fileIndex) '\'];
        if ~exist(resultFileDir, 'dir')
            mkdir(resultFileDir);
        end
        
        for i=1:length(neurons)
            h = neurons(i);
            for j=1:length(epochs)
                maxEpochs = epochs(j);
                resultFileNames = cell(numExecutions,1);
                resultAccs = zeros(numExecutions,1);
                resultEps = zeros(numExecutions,1);
                resultYh = cell(numExecutions,1);
                parfor t=1:numExecutions
                    resultFileName = [resultFileDir sprintf('maxEpochs%d_h%d_t%d', maxEpochs, h, t)];
                    resultFileNames{t} = resultFileName;
                    if exist([resultFileName '.mat'], 'file') == 2
                       continue; 
                    end
                    [ A, B, ~, ~, execEpochs ] = MLPtreina( Xtr, Ytr, Xval, Yval, h, maxEpochs, 2, 0, 1 );
                    Y = MLPsaida( Xtest, A, B );
                    [~,YtestClasses] = max(Ytest,[],2);
                    [~,Ytc] = max(Y,[],2); 
                    [accuracy, ~ ] = multiclassConfusionMatrix( YtestClasses, Ytc, classes ); 
                    fprintf('h = %d\tfile = %d\tt = %d\tacc = %f\tmaxEpochs = %d\tep = %d\n', h, fileIndex, t, accuracy, maxEpochs, execEpochs);
                    resultAccs(t) = accuracy; 
                    resultEps(t) = execEpochs;
                    resultYh{t} = Y;
                end
                if saveData == 1
                    for t=1:numExecutions
                        resultFileName = resultFileNames{t};
                        if exist([resultFileName '.mat'], 'file') == 2
                           continue; 
                        end 
                        acc = resultAccs(t);
                        ep = resultEps(t);
                        Yh = resultYh{t};
                        save(resultFileName, 'acc', 'ep', 'Yh');
                    end 
                end                
            end
        end
    end
    
    plotsDir = [baseResultFileDir '\graficos\'];
    if ~exist(plotsDir, 'dir')
        mkdir(plotsDir);
    end
    for j=1:length(epochs)
        maxEpochs = epochs(j);
        accuracies = cell(length(neurons),1);
        epochsExecuted = cell(length(neurons),1);
        for i=1:length(neurons)
            h = neurons(i);
            accs = zeros(numel(fileNames),numExecutions);
            eps = zeros(numel(fileNames),numExecutions);
            for fileIndex=1:numel(fileNames)
                for t=1:numExecutions
                    load([baseResultFileDir num2str(fileIndex) '\' sprintf('maxEpochs%d_h%d_t%d', maxEpochs, h, t)]);
                    accs(fileIndex,t) = acc;
                    eps(fileIndex,t)= ep;
                end 
            end
            accs = reshape(accs, [numel(accs) 1]);
            eps = reshape(eps, [numel(eps) 1]);
            accuracies{i} = accs;
            epochsExecuted{i} = eps;
        end
        cellAccuracies = accuracies;
        accuracies = zeros(length(neurons),numel(fileNames)*numExecutions);
        cellEpochsExecuted = epochsExecuted;
        epochsExecuted = zeros(length(neurons),numel(fileNames)*numExecutions);
        for i=1:length(neurons)
            accuracies(i,:) = cellAccuracies{i};
            epochsExecuted(i,:) = cellEpochsExecuted{i};
        end        
        meanAccuracies = mean(accuracies,2);
        stdAccuracies = std(accuracies,1,2);
        
        meanEpochsExecuted = mean(epochsExecuted,2);
        
        if ~exist([plotsDir 'neuronios_acuracia'], 'dir')
            mkdir([plotsDir 'neuronios_acuracia']);
        end
        neuronsAccuracyImpactPlotFile = [plotsDir 'neuronios_acuracia\neuronios_acuracia_' num2str(maxEpochs)];
        if exist(neuronsAccuracyImpactPlotFile, 'file') ~= 2
            errorbar(neurons, meanAccuracies, stdAccuracies);
            xlabel('Neuronios');
            ylabel('Acuracia');
            title(sprintf('Qtde. de neuronios na camada oculta X Acuracia (max %d epocas)', maxEpochs));
            if saveData == 1
                print(neuronsAccuracyImpactPlotFile, '-dpng');  
            end
        end
        
        if ~exist([plotsDir 'epocas_acuracia'], 'dir')
            mkdir([plotsDir 'epocas_acuracia']);
        end
        epochsAccuracyImpactPlotFile = [plotsDir 'epocas_acuracia\epocas_acuracia_' num2str(maxEpochs)];
        if exist(epochsAccuracyImpactPlotFile, 'file') ~= 2
            plot(meanEpochsExecuted, meanAccuracies, '-X', 'MarkerSize', 12)
            xlabel('Epocas');
            ylabel('Acuracia');
            title('Epocas X Acuracia');
            if saveData == 1
                print(epochsAccuracyImpactPlotFile, '-dpng');
            end
        end

        if ~exist([plotsDir 'epocas_exec_neuronios'], 'dir')
            mkdir([plotsDir 'epocas_exec_neuronios']);
        end
        epochsExecutedMaxEpochsPlotFile = [plotsDir 'epocas_exec_neuronios\epocas_exec_neuronios_' num2str(maxEpochs)];
        if exist(epochsExecutedMaxEpochsPlotFile, 'file') ~= 2
            plot(neurons, meanEpochsExecuted, '--Xr', 'MarkerSize', 12)
            xlabel('Neuronios');
            ylabel('Epocas executadas');
            title('Neuronios X Epocas executadas');
            if saveData == 1
                print(epochsExecutedMaxEpochsPlotFile, '-dpng');
            end
        end
    end
end

function [Xtr, Ytr, Xtest, Ytest, Xval, Yval] = createDatasets(XtrFolds, YtrFolds, XvalFolds, YvalFolds, k)
    [Xtr, Ytr, Xtest, Ytest] = createFoldDatasets(XtrFolds, YtrFolds, k);
    [Xval, Yval, ~, ~] = createFoldDatasets(XvalFolds, YvalFolds, k);
end

function [X, Y, Xk, Yk] = createFoldDatasets(XFolds, YFolds, k)
    
    Xk = XFolds{k};
    Yk = YFolds{k}; 
  
    X = XFolds;
    Y = YFolds;
    X{k} = [];
    Y{k} = [];
    X = cell2mat(X(:));
    Y = cell2mat(Y(:));
end
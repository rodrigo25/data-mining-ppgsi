function [] = main_spambase_kfold()
    % executa varias parametrizacoes de uma MLP com gradiente descendente e
    % método de bisseção
    % dados intermediarios sao persistidos assim que possivel 
    % sao criados graficos ao final da execucao
    foldCount = 5;
    numExecutions = 10;
    neurons = [5 10 20 50 100];
    epochs = [100 200 500];
    fileNames = pre_processing('data_spambase',0, 'kfold', foldCount,'pca');
    baseResultFileDir = 'grid\spambase\kfold\';
    
    for fileIndex=1:numel(fileNames)
        resultFileDir = [baseResultFileDir num2str(fileIndex) '\'];
        if ~exist(resultFileDir, 'dir')
            mkdir(resultFileDir);
        end    
        data = load(fileNames{1});
        XtrFolds = data.XtrFolds;
        YtrFolds = data.YtrFolds;
        XvalFolds = data.XvalFolds;
        YvalFolds = data.YvalFolds;
        preProcInfo = data.preProcInfo;
        classes = preProcInfo.classes;
        for neuronIndex=1:length(neurons)
            h = neurons(neuronIndex);
            for epochsIndex=1:length(epochs)
               maxEpochs = epochs(epochsIndex);
               trainMLP(h, maxEpochs, fileIndex, numExecutions, resultFileDir, XtrFolds, YtrFolds, XvalFolds, YvalFolds, classes, foldCount); 
            end
        end       
    end
    
    % plots
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
            accs = [];
            eps = [];
            for fileIndex=1:numel(fileNames)
                for t=1:numExecutions
                    load([baseResultFileDir num2str(fileIndex) '\' sprintf('h%d_maxEpochs%d_file%d_t%d_k%d', h, maxEpochs, fileIndex, t, k)]);
                    accs = [accs acc];
                    eps  = [eps ep];
                end 
            end
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

function [] = trainMLP(h, maxEpochs, fileIndex, numExecutions, resultFileDir, XtrFolds, YtrFolds, XvalFolds, YvalFolds, classes, foldCount)
    %parfor (t=1:numExecutions,3)
    for k=1:foldCount
        [Xtr, Ytr, Xtest, Ytest, Xval, Yval] = createDatasets(XtrFolds, YtrFolds, XvalFolds, YvalFolds, k);
        resultFileNames = cell(numExecutions,1);
        resultAccs = zeros(numExecutions,1);
        resultEps = zeros(numExecutions,1);
        resultYh = cell(numExecutions,1);
        tic
        parfor (t=1:numExecutions,4)
            resultFileName = [resultFileDir sprintf('h%d_maxEpochs%d_file%d_t%d_k%d', h, maxEpochs, fileIndex, t, k)];
            resultFileNames{t} = resultFileName;
            if exist([resultFileName '.mat'], 'file') == 2
               continue; 
            end

            [ A, B, ~, ~, execEpochs ] = MLPtreina( Xtr, Ytr, Xval, Yval, h, maxEpochs, 2, 0, 1 );
            %[ A, B, ERROtr, ~, ~ ] = MLP_alfaAdaptativo( Xtr, Ytr, Xval, Yval, h, maxEpochs, 1 );
            %execEpochs = length(ERROtr);

            Y = MLPsaida( Xtest, A, B );
            
            Y(Y >= .5) = 1;
            Y(Y < .5) = 0;

            [accuracy, ~ ] = multiclassConfusionMatrix( Ytest, Y, classes ); 
            fprintf('file = %d\tk = %d\tt = %d\tacc = %f\th = %d\tep = %d\n', fileIndex, k, t, accuracy, h, execEpochs);
            resultAccs(t) = accuracy; 
            resultEps(t) = execEpochs;
            resultYh{t} = Y;
        end 
        toc
        %saves data
        for t=1:numExecutions
            resultFileName = resultFileNames{t};
            if exist([resultFileName '.mat'], 'file') == 2
               continue; 
            end
            acc = resultAccs(t);
            ep = resultEps(t);
            Yh = resultYh{t};
            multiclassConfusionMatrix(Ytest, Yh, classes, 1, 'spambase', [resultFileName '_confusion_matrix'] );
            save(resultFileName, 'acc', 'ep', 'Yh');
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
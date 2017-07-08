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
    baseResultFileDir = 'grid\spambase-3\kfold\';
    
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
    
    % accuracy plots
    mlpPlotNames = {};
    accuracies = {};
    executedEpochs = {};
    mlpPlotIndex = 1;
    for neuronIndex=1:numel(mlpNames)
        h = mlpNames{neuronIndex};
        mlpResultFileDir = [baseResultFileDir h '\'];
        if strcmp(h, 'MLP-alfa-fixed' ) || strcmp(h,'MLP-step-decay')
            for alfaIndex=1:length(alfas)
                alfa = alfas(alfaIndex);
                mlpNameWithAlfa = [h '-' num2str(alfa)];
                mlpAccuracies = [];
                mlpExecutedEpochs = [];
                for fileIndex=1:numel(fileNames)
                    mlpResultFileDir = [baseResultFileDir mlpNameWithAlfa '\' num2str(fileIndex) '\']; 
                    for t=1:numExecutions
                        load([mlpResultFileDir sprintf('%s_file%d_t%d', h, fileIndex, t)]);
                        mlpAccuracies = [mlpAccuracies;acc];
                        mlpExecutedEpochs = [mlpExecutedEpochs;ep];
                    end
                end
                
                mlpPlotNames{1,mlpPlotIndex} = mlpNameWithAlfa;
                accuracies{mlpPlotIndex} = mlpAccuracies;
                executedEpochs{mlpPlotIndex} = mlpExecutedEpochs;
                mlpPlotIndex = mlpPlotIndex + 1;
            end 
        else
            mlpAccuracies = [];
            mlpExecutedEpochs = [];
            for fileIndex=1:numel(fileNames)
                for t=1:numExecutions
                    load([mlpResultFileDir num2str(fileIndex) '\' sprintf('%s_file%d_t%d', h, fileIndex, t)]);
                    mlpAccuracies = [mlpAccuracies;acc];
                    mlpExecutedEpochs = [mlpExecutedEpochs;ep];
                end
            end
            mlpPlotNames{1,mlpPlotIndex} = h;
            accuracies{mlpPlotIndex} = mlpAccuracies;
            executedEpochs{mlpPlotIndex} = mlpExecutedEpochs;
            mlpPlotIndex = mlpPlotIndex + 1;
        end
    end

    
    plotResultFileDir = [baseResultFileDir '\graficos'];
    if ~exist(plotResultFileDir, 'dir')
        mkdir(plotResultFileDir);
    end
    
    % accuracies
    boxplotData = [];
    boxplotGroup = [];
    for neuronIndex=1:numel(mlpPlotNames)
        boxplotData = [boxplotData; accuracies{neuronIndex}];
        boxplotGroup = [boxplotGroup; neuronIndex*ones(size(accuracies{neuronIndex}))];
    end
    figure
    boxplot(boxplotData,boxplotGroup);
    title('Acurácias');
    print([plotResultFileDir '\accuracies_boxplot'], '-dpng');
    
    %epochs executed
    boxplotData = [];
    boxplotGroup = [];
    for neuronIndex=1:numel(mlpPlotNames)
        boxplotData = [boxplotData; executedEpochs{neuronIndex}];
        boxplotGroup = [boxplotGroup; neuronIndex*ones(size(executedEpochs{neuronIndex}))];
    end
    figure
    boxplot(boxplotData,boxplotGroup);
    title('Épocas executadas');
    print([plotResultFileDir '\executed_epochs_boxplot'], '-dpng');
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
        for t=1:numExecutions
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
            fprintf('file = %d\tk = %d\tt = %d\tacc = %f\tep = %d\n', fileIndex, k, t, accuracy, execEpochs);
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
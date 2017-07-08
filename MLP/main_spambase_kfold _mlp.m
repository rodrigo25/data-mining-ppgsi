function [] = main_spambase_kfold_mlp()
    
    foldCount = 5;
    numExecutions = 10;
    mlpNames = {'MLP-alfa-fixed' 'MLP-step-decay' 'MLP-bisect' 'MLP-bisec-grad-conj' 'MLP-adaptative'};
    alfas = [0.9 0.5 0.2 0.1 0.01];
    fileNames = pre_processing('data_spambase',0, 'kfold', foldCount);
    baseResultFileDir = 'mlp\spambase\kfold\';
    for fileIndex=1:numel(fileNames)
        data = load(fileNames{1});
        XtrFolds = data.XtrFolds;
        YtrFolds = data.YtrFolds;
        XvalFolds = data.XvalFolds;
        YvalFolds = data.YvalFolds;
        preProcInfo = data.preProcInfo;
        classes = preProcInfo.classes;
        for mlpIndex=1:numel(mlpNames)
            mlpName = mlpNames{mlpIndex};
            mlpResultFileDir = [baseResultFileDir mlpName '\'];
            if strcmp(mlpName, 'MLP-alfa-fixed' ) || strcmp(mlpName,'MLP-step-decay')
                for alfaIndex=1:length(alfas)
                    alfa = alfas(alfaIndex);
                    mlpResultFileDir = [baseResultFileDir mlpName '-' num2str(alfa) '\'];
                    
                    resultFileDir = [mlpResultFileDir num2str(fileIndex) '\'];
                    if ~exist(resultFileDir, 'dir')
                        mkdir(resultFileDir);
                    end
                    
                    trainMLP(mlpName, fileIndex, numExecutions, resultFileDir, XtrFolds, YtrFolds, XvalFolds, YvalFolds, classes, foldCount, alfa);
                end
            else
                resultFileDir = [mlpResultFileDir num2str(fileIndex) '\'];
                if ~exist(resultFileDir, 'dir')
                    mkdir(resultFileDir);
                end
                    
                trainMLP(mlpName, fileIndex, numExecutions, resultFileDir, XtrFolds, YtrFolds, XvalFolds, YvalFolds, classes, foldCount);
            end            
        end       
    end
    
    % accuracy plots
    mlpPlotNames = {};
    accuracies = {};
    executedEpochs = {};
    mlpPlotIndex = 1;
    for mlpIndex=1:numel(mlpNames)
        mlpName = mlpNames{mlpIndex};
        mlpResultFileDir = [baseResultFileDir mlpName '\'];
        if strcmp(mlpName, 'MLP-alfa-fixed' ) || strcmp(mlpName,'MLP-step-decay')
            for alfaIndex=1:length(alfas)
                alfa = alfas(alfaIndex);
                mlpNameWithAlfa = [mlpName '-' num2str(alfa)];
                mlpAccuracies = [];
                mlpExecutedEpochs = [];
                for fileIndex=1:numel(fileNames)
                    mlpResultFileDir = [baseResultFileDir mlpNameWithAlfa '\' num2str(fileIndex) '\']; 
                    for t=1:numExecutions
                        load([mlpResultFileDir sprintf('%s_file%d_t%d', mlpName, fileIndex, t)]);
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
                    load([mlpResultFileDir num2str(fileIndex) '\' sprintf('%s_file%d_t%d', mlpName, fileIndex, t)]);
                    mlpAccuracies = [mlpAccuracies;acc];
                    mlpExecutedEpochs = [mlpExecutedEpochs;ep];
                end
            end
            mlpPlotNames{1,mlpPlotIndex} = mlpName;
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
    for mlpIndex=1:numel(mlpPlotNames)
        boxplotData = [boxplotData; accuracies{mlpIndex}];
        boxplotGroup = [boxplotGroup; mlpIndex*ones(size(accuracies{mlpIndex}))];
    end
    figure
    boxplot(boxplotData,boxplotGroup);
    title('Acurácias');
    print([plotResultFileDir '\accuracies_boxplot'], '-dpng');
    
    %epochs executed
    boxplotData = [];
    boxplotGroup = [];
    for mlpIndex=1:numel(mlpPlotNames)
        boxplotData = [boxplotData; executedEpochs{mlpIndex}];
        boxplotGroup = [boxplotGroup; mlpIndex*ones(size(executedEpochs{mlpIndex}))];
    end
    figure
    boxplot(boxplotData,boxplotGroup);
    title('Épocas executadas');
    print([plotResultFileDir '\executed_epochs_boxplot'], '-dpng');
end

function [] = trainMLP(mlpName, fileIndex, numExecutions, resultFileDir, XtrFolds, YtrFolds, XvalFolds, YvalFolds, classes, foldCount, alfa)
    resultFileNames = cell(numExecutions,1);
    resultAccs = zeros(numExecutions,1);
    resultEps = zeros(numExecutions,1);
    resultYh = cell(numExecutions,1);
    %parfor (t=1:numExecutions,3)
    for k=1:foldCount
        [Xtr, Ytr, Xtest, Ytest, Xval, Yval] = createDatasets(XtrFolds, YtrFolds, XvalFolds, YvalFolds, k);
        [~,YtestClasses] = max(Ytest,[],2);
        for t=1:numExecutions
            resultFileName = [resultFileDir sprintf('%s_file%d_t%d_k%d', mlpName, fileIndex, t, k)];
            resultFileNames{t} = resultFileName;
            if exist([resultFileName '.mat'], 'file') == 2
               continue; 
            end

            switch mlpName
                case 'MLP-alfa-fixed'
                    [ A, B, ~, ~, execEpochs ] = MLPtreina( Xtr, Ytr, Xval, Yval, 20, 1000, 0, 0, 1, alfa );
                case 'MLP-step-decay'
                    [ A, B, ~, ~, execEpochs ] = MLPtreina( Xtr, Ytr, Xval, Yval, 20, 1000, 1, 0, 1, alfa );
                case 'MLP-bisect' 
                    [ A, B, ~, ~, execEpochs ] = MLPtreina( Xtr, Ytr, Xval, Yval, 20, 1000, 2, 0, 1 );
                case 'MLP-bisec-grad-conj' 
                    [ A, B, ~, ~, execEpochs ] = MLPtreina( Xtr, Ytr, Xval, Yval, 20, 1000, 2, 1, 1 );
                case 'MLP-adaptative'
                    [ A, B, ERROtr, ~, ~ ] = MLP_alfaAdaptativo( Xtr, Ytr, Xval, Yval, 20, 1000, 1 );
                    execEpochs = length(ERROtr);
                otherwise
                    error('dont know');
            end        

            Y = MLPsaida( Xtest, A, B );

            [~,Ytc] = max(Y,[],2); 
            [accuracy, ~ ] = multiclassConfusionMatrix( YtestClasses, Ytc, classes ); 
            fprintf('MLP = %s\tfile = %d\tk = %d\tt = %d\tacc = %f\tep = %d\n', mlpName, fileIndex, k, t, accuracy, execEpochs);
            resultAccs(t) = accuracy; 
            resultEps(t) = execEpochs;
            resultYh{t} = Y;
        end 
    end
    %saves data
    for t=1:numExecutions
        resultFileName = resultFileNames{t};
        if exist([resultFileName '.mat'], 'file') == 2
           continue; 
        end
        acc = resultAccs(t);
        ep = resultEps(t);
        Yh = resultYh{t};
        [~,Yhc] = max(Yh,[],2);
        confusionMatrixName = mlpName;
        if strcmp(mlpName,'MLP-alfa-fixed') || strcmp(mlpName, 'MLP-step-decay')
            confusionMatrixName = [mlpName ' alfa' num2str(alfa)];
        end
        multiclassConfusionMatrix(YtestClasses, Yhc, classes, 1, confusionMatrixName, [resultFileName '_confusion_matrix'] );
        
        save(resultFileName, 'acc', 'ep', 'Yh');
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
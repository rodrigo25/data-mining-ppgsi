function [] = main_spambase_kfold_roc()
    
    foldCount = 5;
    numExecutions = 10;
    neurons = [5 100]; % melhor e pior da ultima etapa
    epochs = [500];
    fileNames = pre_processing('data_spambase',0, 'kfold', foldCount,'pca');
    baseResultFileDir = 'grid\spambase\kfold\roc\';
    
    % treina mlps sobre os folds e persiste as hipoteses de cada uma
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
    
    % le todos os arquivos resultantes e armazena todos os resultados em
    % resultYh
    resultYh = cell(length(neurons),1);
    for i=1:length(neurons)
        h = neurons(i);
        for fileIndex=1:numel(fileNames)
            for t=1:numExecutions
                for k=1:foldCount
                    data = load([baseResultFileDir num2str(fileIndex) '\' sprintf('h%d_maxEpochs%d_file%d_t%d_k%d', h, 500, fileIndex, t, k)]);
                    neuronsYh = resultYh{i};
                    neuronsYh = [neuronsYh data.Yh];
                    resultYh{i} = neuronsYh;
                end
            end 
        end
    end
end

function [] = trainMLP(h, maxEpochs, fileIndex, numExecutions, resultFileDir, XtrFolds, YtrFolds, XvalFolds, YvalFolds, classes, foldCount)
    %parfor (t=1:numExecutions,3)
    for k=1:foldCount
        [Xtr, Ytr, Xtest, ~, Xval, Yval] = createDatasets(XtrFolds, YtrFolds, XvalFolds, YvalFolds, k);
        resultFileNames = cell(numExecutions,1);
        resultYh = cell(numExecutions,1);
        parfor (t=1:numExecutions,4)
            resultFileName = [resultFileDir sprintf('h%d_maxEpochs%d_file%d_t%d_k%d', h, maxEpochs, fileIndex, t, k)];
            resultFileNames{t} = resultFileName;
            if exist([resultFileName '.mat'], 'file') == 2
               continue; 
            end

            [ A, B, ~, ~, ~ ] = MLPtreina( Xtr, Ytr, Xval, Yval, h, maxEpochs, 2, 0, 1 );
            Yh = MLPsaida( Xtest, A, B );
            
            fprintf('file = %d\tk = %d\tt = %d\th = %d\n', fileIndex, k, t, h);
            resultYh{t} = Yh;
        end 
        %saves data
        for t=1:numExecutions
            resultFileName = resultFileNames{t};
            if exist([resultFileName '.mat'], 'file') == 2
               continue; 
            end
            Yh = resultYh{t};
            save(resultFileName, 'Yh');
        end 
    end
end
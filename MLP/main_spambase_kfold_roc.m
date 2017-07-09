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
    
    plotsDir = [baseResultFileDir '\graficos\'];
    if ~exist(plotsDir, 'dir')
        mkdir(plotsDir);
    end
    
    % le todos os arquivos resultantes e armazena todos os resultados em
    % resultYh
    resultY = cell(length(neurons),2);
    for i=1:length(neurons)
        h = neurons(i);
        for fileIndex=1:numel(fileNames)
            for t=1:numExecutions
                for k=1:foldCount
                    data = load([baseResultFileDir num2str(fileIndex) '\' sprintf('h%d_maxEpochs%d_file%d_t%d_k%d', h, 500, fileIndex, t, k)]);
                    
                    neuronsYh = resultY{i,1};
                    neuronsYh = [neuronsYh data.Yh];
                    resultY{i,1} = neuronsYh;
                    
                    neuronsYtest = resultY{i,2};
                    neuronsYtest = [neuronsYtest data.Ytest];
                    resultY{i,2} = neuronsYtest;
                end
            end 
        end
    end
    
    % inicio do codigo de analise roc
    % para cada execucao feita
    for i=1:length(neurons)
        neuronsResultYh = resultY{i,1};
        neuronsResultYtest = resultY{i,2};
        for j=1:size(neuronsResultYh,2)
            
            %DEFINE LIMIAR DE DECISAO
            Ymlp = neuronsResultYh(:,j);
            Yd = neuronsResultYtest(:,j);
            
            for threshold=0:0.005:1
                fN = fN +1;

                Y(Ymlp>=threshold) = 1;
                Y(Ymlp<threshold) = 0;

                %CALCULA RESULTADOS
                [ ~, ~, res_struct] = calc_result( Yd, Y );

                %ARMAZENA TPR E FPR PARA MONTAR A CURVA ROC
                TPR = [TPR; res_struct(2).TPR];
                FPR = [FPR; res_struct(2).FPR];

                im_roc = figure; %cria o grafico ROC
                plot([0 1],[0 1],'b','DisplayName','Diagonal'); %plota a diagonal ascendente
                hold on
                h = neurons(i);
                plot( FPR, TPR, 'DisplayName',['MLP bisseção h=' num2str(h) ' threshold=' num2str(threshold)]); %Plota a curva ROC do componente
                ylabel('True Positive Rate');
                xlabel('False Positive Rate');
                legend('show')
                title('ROC')
                print(im_roc,[plotsDir 'ROC_h=' num2str(h) '_threshold=' num2str(threshold) '_j=' num2str(j)],'-dpng');
            end 
        end
    end
end

function [] = trainMLP(h, maxEpochs, fileIndex, numExecutions, resultFileDir, XtrFolds, YtrFolds, XvalFolds, YvalFolds, classes, foldCount)
    %parfor (t=1:numExecutions,3)
    for k=1:foldCount
        [Xtr, Ytr, Xtest, Ytest, Xval, Yval] = createDatasets(XtrFolds, YtrFolds, XvalFolds, YvalFolds, k);
        resultFileNames = cell(numExecutions,1);
        resultYh = cell(numExecutions,1);
        resultYtest = cell(numExecutions,1);
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
            resultYtest{t} = Ytest;
        end 
        %saves data
        for t=1:numExecutions
            resultFileName = resultFileNames{t};
            if exist([resultFileName '.mat'], 'file') == 2
               continue; 
            end
            Yh = resultYh{t};
            Ytest = resultYtest{t};
            save(resultFileName, 'Yh', 'Ytest');
        end 
    end
end
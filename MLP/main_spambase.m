function [accuracies, epochsExecuted] = main_spambase()

    % preprocessa spambase
    % 0 = nao sobrescreve se existir
    fileName = pre_processing('data_spambase',0);
    load(fileName);
   
    %Varia primeiro apenas neuronios
    K = preProcInfo.k;
    neurons = [2 5 10 20 50 100];
    epochs = 100;
    
    resultFileDir = ['grid' '\' 'spambase' '\' 'kfold' '\'];
    if ~exist(resultFileDir, 'dir')
        mkdir(resultFileDir);
        for i=1:length(neurons)
            h = neurons(i);
            for k=1:K
                [Xtr, Ytr, Xtest, Ytest, Xval, Yval] = createDatasets(XtrFolds, YtrFolds, XvalFolds, YvalFolds, k);
                for t=1:30
                    [ A, B, ~, ~, ep ] = MLPtreina( Xtr, Ytr, Xval, Yval, h, epochs, .01, 0, 1 ); 
                    Y = MLPsaida( Xtest, A, B );
                    Y(Y>=.5) = 1;
                    Y(Y<.5) = 0;
                    [ ~, ~, stats] = calc_result( Ytest, Y );
                    [~,acc] = stats.ACC;
                    fprintf('h = %d\tk = %d\tt = %d\tacc = %f\tep=%d\n', h, k, t, acc, ep);
                    save([resultFileDir sprintf('h%d_k%d_t%d', h, k, t)], 'acc', 'ep');
                end 
            end
        end
    end

    plotsDir = [resultFileDir '\graficos\'];
    if ~exist(plotsDir, 'dir')
        mkdir(plotsDir);
    end

    accuracies = zeros(length(neurons),1);
    epochsExecuted = zeros(length(neurons),1);
    for i=1:length(neurons)
        h = neurons(i);
        totalAcc = 0;
        totalEp = 0;
        for k=1:K
            for t=1:30
                load([resultFileDir sprintf('h%d_k%d_t%d', h, k, t)]);
                totalAcc = totalAcc + acc;
                totalEp = totalEp + ep;
            end 
        end
        totalAcc = totalAcc / 30;
        totalAcc = totalAcc / K;

        totalEp = totalEp / 30;
        totalEp = totalEp / K;
        accuracies(i) = totalAcc;
        epochsExecuted(i) = totalEp;
    end
    neuronsAccuracyImpactPlotFile = [plotsDir 'neuronios_acuracia'];
    if exist(neuronsAccuracyImpactPlotFile, 'file') ~= 2
        plot(neurons, accuracies);
        xlabel('Neuronios');
        ylabel('Acuracia');
        title('Qtde. de neuronios na camada oculta X Acuracia');
        print(neuronsAccuracyImpactPlotFile, '-dpng');
    end
    
    %Avaliados os neuronios, tenta aumentar a acuracia dando ao algoritmo
    %mais epocas de treinamento
    [~,index] = max(accuracies);
    maxAccNeurons = neurons(index);
    epochs = [100 250 500 750 1000];
    
    resultFileDir = ['grid' '\' 'spambase' '\' 'kfold' '\' 'epochs' '\'];
    if ~exist(resultFileDir, 'dir')
        mkdir(resultFileDir);
    end
    for i=1:length(epochs)
        h = maxAccNeurons;
        maxEpochs = epochs(i);
        for k=1:K
            [Xtr, Ytr, Xtest, Ytest, Xval, Yval] = createDatasets(XtrFolds, YtrFolds, XvalFolds, YvalFolds, k);
            for t=1:30
                resultFileName = [resultFileDir sprintf('maxEpochs%d_k%d_t%d', maxEpochs, k, t)];
                if exist([resultFileName '.mat'],'file') == 2
                    continue;
                end
                [ A, B, ~, ~, ep ] = MLPtreina( Xtr, Ytr, Xval, Yval, h, maxEpochs, .01, 0, 1 ); 
                Y = MLPsaida( Xtest, A, B );
                Y(Y>=.5) = 1;
                Y(Y<.5) = 0;
                [ ~, ~, stats] = calc_result( Ytest, Y );
                [~,acc] = stats.ACC;
                fprintf('h = %d\tk = %d\tt = %d\tacc = %f\tep = %d\tmaxEpochs = %d\n', h, k, t, acc, ep, maxEpochs);
                save(resultFileName, 'acc', 'ep');
            end 
        end
    end
        
    accuracies = zeros(length(epochs),1);
    epochsExecuted = zeros(length(epochs),1);
    for i=1:length(epochs)
        h = maxAccNeurons;
        maxEpochs = epochs(i);
        totalAcc = 0;
        totalEp = 0;
        for k=1:K
            for t=1:30
                load([resultFileDir sprintf('maxEpochs%d_k%d_t%d', maxEpochs, k, t)]);
                totalAcc = totalAcc + acc;
                totalEp = totalEp + ep;
            end 
        end
        totalAcc = totalAcc / 30;
        totalAcc = totalAcc / K;

        totalEp = totalEp / 30;
        totalEp = totalEp / K;
        accuracies(i) = totalAcc;
        epochsExecuted(i) = totalEp;
    end
    epochsAccuracyImpactPlotFile = [plotsDir 'epocas_acuracia'];
    if exist(epochsAccuracyImpactPlotFile, 'file') ~= 2
        plot(epochsExecuted, accuracies, '-X', 'MarkerSize', 12)
        xlabel('Epocas');
        ylabel('Acuracia');
        title('Epocas X Acuracia');
        print(epochsAccuracyImpactPlotFile, '-dpng');
    end
    
    epochsExecutedMaxEpochsPlotFile = [plotsDir 'epocas_exec_max_epocas'];
    if exist(epochsExecutedMaxEpochsPlotFile, 'file') ~= 2
        plot(epochs, epochsExecuted, '--Xr', 'MarkerSize', 12)
        xlabel('Maximos de epocas');
        ylabel('Epocas executadas');
        title('Maxs epocas X Epocas executadas');
        print(epochsExecutedMaxEpochsPlotFile, '-dpng');
    end
    
    
    %Avaliados os neuronios e epocas, varia tratamento da taxa de
    %aprendizado
    maxAccNeurons = 10;
    maxEpochs = 500;
    alfaApproaches = [0 1 2 3];
    
    resultFileDir = ['grid' '\' 'spambase' '\' 'kfold' '\' 'alfas' '\'];
    if ~exist(resultFileDir, 'dir')
        mkdir(resultFileDir);
    end
    for i=1:length(alfaApproaches)
        h = maxAccNeurons;
        alfaApproach = alfaApproaches(i);
        for k=1:K
            [Xtr, Ytr, Xtest, Ytest, Xval, Yval] = createDatasets(XtrFolds, YtrFolds, XvalFolds, YvalFolds, k);
            for t=1:30
                resultFileName = [resultFileDir sprintf('maxEpochs%d_k%d_t%d_alfa%d', maxEpochs, k, t, alfaApproach)];
                if exist([resultFileName '.mat'],'file') == 2
                    continue;
                end
                
                if alfaApproach==3
                    [ A, B ] = treina_rede( Xtr, Ytr, h, maxEpochs );
                else
                    [ A, B, ~, ~, ~ ] = MLPtreina( Xtr, Ytr, Xval, Yval, h, maxEpochs, .01, alfaApproach, 1 ); 
                end

                Y = MLPsaida( Xtest, A, B );
                Y(Y>=.5) = 1;
                Y(Y<.5) = 0;
                [ ~, ~, stats] = calc_result( Ytest, Y );
                [~,acc] = stats.ACC;
                fprintf('h = %d\tk = %d\tt = %d\tacc = %f\tep = %d\tmaxEpochs = %d\talfaApproach=%d\n', h, k, t, acc, ep, maxEpochs, alfaApproach);
                save(resultFileName, 'acc', 'ep');
            end 
        end
    end
        
    accuracies = zeros(length(alfaApproaches),1);
    epochsExecuted = zeros(length(alfaApproaches),1);
    maxEpochs = 500;
    for i=1:length(alfaApproaches)
        alfaApproach = alfaApproaches(i);
        totalAcc = 0;
        totalEp = 0;
        for k=1:K
            for t=1:30
                load([resultFileDir sprintf('maxEpochs%d_k%d_t%d_alfa%d', maxEpochs, k, t, alfaApproach)]);
                totalAcc = totalAcc + acc;
                totalEp = totalEp + ep;
            end 
        end
        totalAcc = totalAcc / 30;
        totalAcc = totalAcc / K;

        totalEp = totalEp / 30;
        totalEp = totalEp / K;
        accuracies(i) = totalAcc;
        epochsExecuted(i) = totalEp;
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
function [] = main_winequality()
    load('data_winequality-red.mat');
    %load('data_winequality-white.mat');

    % preprocessa data_winequality-red
    % 0 = nao sobrescreve se existir
    fileName = pre_processing('data_winequality-red',0);
    load(fileName);
   
    K = preProcInfo.k;
    neurons = 10:10:100;
    epochs = 500;

    resultFileDir = ['grid' '\' 'winequality' '\' 'red' '\' 'kfold' '\'];
    if ~exist(resultFileDir, 'dir')
        mkdir(resultFileDir);
        for i=1:length(neurons)
            h = neurons(i);
            maxEpochs = epochs;

            for k=1:K 
                [Xtr, Ytr, Xtest, Ytest, Xval, Yval] = createDatasets(XtrFolds, YtrFolds, XvalFolds, YvalFolds, k);
                for t=1:30
                    resultFileName = [resultFileDir sprintf('h%d_k%d_t%d', h, k, t)];
                    if exist([resultFileName '.mat'], 'file') == 2
                       continue; 
                    end
                    fprintf('h = %d epochs = %d alfaApproach=%d k=%d t=%d\n', h, maxEpochs, alfaApproach, k, t);
                    [ A, B, ~, ~, ep ] = MLPtreina( Xtr, Ytr, Xval, Yval, h, maxEpochs, .01, 0, 1 );
                    Yc = MLPsaida( Xtest, A, B );
                    [~,Ytc] = max(Ytest,[],2);
                    [~,Yc] = max(Yc,[],2); 
                    [acc, ~ ] = multiclassConfusionMatrix( Ytc, Yc, classes ); 
                    fprintf('h = %d\tk = %d\tt = %d\tacc = %f\tep=%d\n', h, k, t, acc, ep);
                    save(resultFileName, 'acc', 'ep');
                end
            end
        end
    end

    plotsDir = [resultFileDir '\graficos\'];
    if ~exist(plotsDir, 'dir')
        mkdir(plotsDir);
    end
    neuronsAccuracyImpactPlotFile = [plotsDir 'neuronios_acuracia'];
    if exist(neuronsAccuracyImpactPlotFile, 'file') ~= 2
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

        plot(neurons, accuracies);
        xlabel('Neuronios');
        ylabel('Acuracia');
        title('Qtde. de neuronios na camada oculta X Acuracia');
        print(neuronsAccuracyImpactPlotFile, '-dpng'); 
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
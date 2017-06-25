function [accuracies, epochsExecuted] = main_winequality()
    load('data_winequality-red.mat');
    %load('data_winequality-white.mat');

    % preprocessa data_winequality-red
    % 0 = nao sobrescreve se existir
    fileName = pre_processing('data_winequality-red',0);
    load(fileName);
   
    K = preProcInfo.k;
    classes = preProcInfo.classes;
    neurons = [3 5 7 10:10:100];
    epochs = [200 500 1000 2000];

    resultFileDir = ['grid' '\' 'winequality' '\' 'red' '\' 'kfold' '\'];
    if ~exist(resultFileDir, 'dir')
        mkdir(resultFileDir);
    end
    for i=1:length(neurons)
        h = neurons(i);
        for j=1:length(epochs)
            maxEpochs = epochs(j);
            for k=1:K 
                [Xtr, Ytr, Xtest, Ytest, Xval, Yval] = createDatasets(XtrFolds, YtrFolds, XvalFolds, YvalFolds, k);
                for t=1:30
                    resultFileName = [resultFileDir sprintf('maxEpochs%d_h%d_k%d_t%d', maxEpochs, h, k, t)];
                    if exist([resultFileName '.mat'], 'file') == 2
                       continue; 
                    end
                    [ A, B, ~, ~, ep ] = MLPtreina( Xtr, Ytr, Xval, Yval, h, maxEpochs, .01, 0, 1 );
                    Yc = MLPsaida( Xtest, A, B );
                    [~,Ytc] = max(Ytest,[],2);
                    [~,Yc] = max(Yc,[],2); 
                    [acc, ~ ] = multiclassConfusionMatrix( Ytc, Yc, classes ); 
                    fprintf('h = %d\tk = %d\tt = %d\tacc = %f\tmaxEpochs = %d\tep = %d\n', h, k, t, acc, maxEpochs, ep);
                    save(resultFileName, 'acc', 'ep');
                end
            end 
        end
    end

    plotsDir = [resultFileDir '\graficos\'];
    if ~exist(plotsDir, 'dir')
        mkdir(plotsDir);
    end
    for j=1:length(epochs)
        maxEpochs = epochs(j);
        accuracies = zeros(length(neurons),1);
        epochsExecuted = zeros(length(neurons),1);
        for i=1:length(neurons)
            h = neurons(i);
            totalAcc = 0;
            totalEp = 0;
            for k=1:K
                for t=1:30
                    load([resultFileDir sprintf('maxEpochs%d_h%d_k%d_t%d', maxEpochs, h, k, t)]);
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
        
        neuronsAccuracyImpactPlotFile = [plotsDir 'neuronios_acuracia_' maxEpochs];
        if exist(neuronsAccuracyImpactPlotFile, 'file') ~= 2
            plot(neurons, accuracies);
            xlabel('Neuronios');
            ylabel('Acuracia');
            title(sprintf('Qtde. de neuronios na camada oculta X Acuracia (max %d epocas)', maxEpochs));
            print(neuronsAccuracyImpactPlotFile, '-dpng'); 
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
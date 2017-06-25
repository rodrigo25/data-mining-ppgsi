function [accuracies, epochsExecuted] = main_spambase()

    % preprocessa spambase
    % 0 = nao sobrescreve se existir
    fileName = pre_processing('data_spambase',0);
    load(fileName);
   
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
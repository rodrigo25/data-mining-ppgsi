function [] = main_winequality()
    load('data_winequality-red.mat');
    %load('data_winequality-white.mat');

    N = size(X,1);

    % Z-SCORE NORALIZATION
    X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1); 

    [Yd, classes] = multiclassY(Y);

    % SEPARACAO CONJ DE TREINAMENTO E TESTE
    variablesFileName = 'winequality_red_kfold.mat';
    if exist(variablesFileName,'file') == 2
        load(variablesFileName);
    else
        % holdout
        %[ Xtr, Ytr, Xtest, Ytest ] = holdout( X, Y, 0.7 );

        % k-fold
         K =5;
        [ XtrFolds, YtrFolds ] = kfoldCV( X, Yd, K );
        %save('winequality_red_holdout', 'Xtr', 'Ytr', 'Xtest', 'Ytest'); 
        XvalFolds = cell(K);
        YvalFolds = cell(K);
        for i=1:K
            Xtr = XtrFolds{i};
            Ytr = YtrFolds{i};
            [ Xtr, Ytr, Xval, Yval ] = holdout( Xtr, Ytr, 0.66 );
            XtrFolds{i} = Xtr;
            YtrFolds{i} = Ytr;
            XvalFolds{i} = Xval;
            YvalFolds{i} = Yval;
        end
        save('winequality_red_kfold', 'K', 'XtrFolds', 'YtrFolds', 'XvalFolds', 'YvalFolds');
    end

    neurons = 10:5:80;
    epochs = [500];

    alfaApproaches = [0 1 2]; %0 fixed alfa,  1 - step-decay,  2 validation-error-decay

    accuracies = zeros(length(neurons),length(epochs));
    epochsExecuted = zeros(length(neurons),length(epochs));

    for i=1:length(neurons)
        for j=1:length(epochs)
            for a=1:length(alfaApproaches)
                h = neurons(i);
                maxEpochs = epochs(j);
                alfaApproach = alfaApproaches(a);

                resultFileDir = ['winequality' '\' 'red' '\' 'kfold' '\'];
                if ~exist(resultFileDir, 'dir')
                    mkdir(resultFileDir);
                end
                resultFileName = [resultFileDir sprintf('h%d_maxEpochs%d_alfa%d', h, maxEpochs, alfaApproach)];
                if exist([resultFileName '.mat'], 'file') == 2
                   continue; 
                end
                
                totalAcc = 0;
                totalEpochs = 0;
                for k=1:K 
                    [Xtr, Ytr, Xtest, Ytest, Xval, Yval] = createDatasets(XtrFolds, YtrFolds, XvalFolds, YvalFolds, k);
                    for t=1:30
                        fprintf('h = %d epochs = %d alfaApproach=%d k=%d t=%d\n', h, maxEpochs, alfaApproach, k, t);
                        if alfaApproach==0 %fixed
                            % TREINA MLP
                            [ A, B, ~, ~, ep ] = MLPtreina( Xtr, Ytr, Xval, Yval, h, maxEpochs, .01, 0, 1 );
                        elseif alfaApproach==1 %step decay
                            [ A, B, ~, ~, ep ] = MLPtreina( Xtr, Ytr, Xval, Yval, h, maxEpochs, .2, 1, 1 );
                        elseif alfaApproach==2 %validation based decay
                            [ A, B, ~, ~, ep ] = MLPtreina( Xtr, Ytr, Xval, Yval, h, maxEpochs, .2, 2, 1 );
                        end

                        % CALCULA SAIDA
                        Yc = MLPsaida( Xtest, A, B );

                        [~,Ytc] = max(Ytest,[],2);
                        [~,Yc] = max(Yc,[],2); 
                        
                        [acc, ~ ] = multiclassConfusionMatrix( Ytc, Yc, classes ); 
                        totalAcc = totalAcc + acc;
                        totalEpochs = totalEpochs + ep;
                    end
                    
                    accuracies(i,j) = accuracies(i,j) + ( totalAcc / 30);
                    epochsExecuted(i,j) = epochsExecuted(i,j) + ( totalEpochs / 30); 
                end
                
                foldsMeanAcc = accuracies(i,j) / K;
                foldsMeanEpochs = epochsExecuted(i,j) / K;
                accuracies(i,j) = foldsMeanAcc;      
                epochsExecuted(i,j) = foldsMeanEpochs;
                save(resultFileName, 'foldsMeanAcc', 'foldsMeanEpochs');
            end
        end
    end

    save(['winequality' '\' 'red' '\' 'kfold' '\' 'means'], 'accuracies', 'epochsExecuted');
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
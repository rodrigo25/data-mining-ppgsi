function [filesOut] = pre_processing(dataset,shouldOverwrite,typeCrossValidation,count,reductionType)
    if nargin < 5
        reductionType = 'none';
        if nargin < 4
            count = 1;
            if nargin < 3
                %typeCrossValidation = 'holdout';
                typeCrossValidation = 'kfold';        
                if nargin < 2
                    shouldOverwrite = -1;
                    if nargin < 1
                        %SELECIONA DATASET
                        datasetOpt = input('Selecione o dataset [1]data_spambase [2]data_winequality-red [3]data_winequality-white: ');

                        switch datasetOpt
                            case 1
                                dataset = 'data_spambase';
                            case 2
                                dataset = 'data_winequality-red';
                            case 3
                                dataset = 'data_winequality-white';
                            otherwise
                                error('Unknown dataset option');
                        end
                    end
                end 
            end
        end
    end
    %CARREGA DATASET
    load(['data/' dataset '.mat']);

    %NORMALIZACAO DOS DADOS
    typeNormalization = 'zscore';
    %typeNormalization = 'min-max';

    [X, mean_val, std_val,min_val, max_val] = normalization( X, typeNormalization );
    
    if strcmp(reductionType,'pca')
       [ coefs, ~, pcvar ] = pca(X);    
        relVar = pcvar / sum(pcvar);
        for dim=1:length(relVar)
            if sum(relVar(1:dim)) >= 0.95
                break;
            end
        end 
        X = X * coefs(:,1:dim);
    end

    %TRATANDO SAIDA DE PROBLEMAS MULTICLASS
    classes = unique(Y);
    if length(classes)>2
      [Y] = multiclassY(Y);
    end

    %CROSS-VALIDATION
    %typeCrossValidation = 'holdout';
    %typeCrossValidation = 'kfold';

    %informacoes sobre o pre-processamento
    preProcInfo = struct('dataset',dataset, ...
        'classes',classes,...
        'crossValidation',typeCrossValidation,'k',[], ...
        'normalization',typeNormalization, ...
        'mean_val', mean_val, ...
        'std_val',std_val,...
        'min_val',min_val,'max_val',max_val);
    
    disp(preProcInfo);

    filesOut = cell(count,1);
    for i=1:count
        if strcmp(typeCrossValidation,'holdout')
            % HOLDOUT  
            k = 0.66; %porcentagem de dados de treinamento
            [ Xtr, Ytr, Xtest, Ytest ] = holdout( X, Y, k ); %k-fold cross-validation
            [ Xtr, Ytr, Xval, Yval ] = holdout( Xtr, Ytr, 0.66 );

            preProcInfo.k = k;

            fileOut = ['data/' dataset '_holdout' num2str(i)]; %define o nome do arq de saida
            filesOut{i} = fileOut;
            
            if exist([fileOut '.mat'], 'file') %verifica se o arquivo ja existe
                if shouldOverwrite == -1
                    choice = questdlg('The file allready exist, do you want to overwrite it?', 'Warning', 'Yes','No','No');
                    if strcmp(choice,'No')
                        shouldOverwrite = 0;
                    else
                        shouldOverwrite = 1;
                    end
                end
                if shouldOverwrite == 0 %dont overwrite
                    continue; 
                end  
            end

            save(fileOut,'Xtr','Ytr','Xval','Yval','Xtest','Ytest','preProcInfo'); %salva arq 

        elseif strcmp(typeCrossValidation,'kfold')
            % K-FOLD
            K =5; %numero de folds
            [ XtrFolds, YtrFolds ] = kfoldCV( X, Y, K );

            preProcInfo.k = K;

            fileOut = ['data/' dataset '_kfold_k' num2str(K) '_' num2str(i)]; %define o nome do arq de saida
            filesOut{i} = fileOut;
            
            XvalFolds = cell(K,1);
            YvalFolds = cell(K,1);
            for k=1:K
              Xtr = XtrFolds{k};
              Ytr = YtrFolds{k};
              [ Xtr, Ytr, Xval, Yval ] = holdout( Xtr, Ytr, 0.66 );
              XtrFolds{k} = Xtr;
              YtrFolds{k} = Ytr;
              XvalFolds{k} = Xval;
              YvalFolds{k} = Yval;
            end

            if exist([fileOut '.mat'], 'file') && shouldOverwrite == 0%verifica se o arquivo ja existe
                if shouldOverwrite == -1
                    choice = questdlg('The file already exist, do you want to overwrite it?', 'Warning', 'Yes','No','No');
                    if strcmp(choice,'No')
                        shouldOverwrite = 0;
                    else
                        shouldOverwrite = 1;
                    end
                end          
                if shouldOverwrite == 0
                    return; 
                end
            end
            save(fileOut,'XtrFolds','YtrFolds','XvalFolds','YvalFolds','preProcInfo'); %salva arq
        end
    end
end
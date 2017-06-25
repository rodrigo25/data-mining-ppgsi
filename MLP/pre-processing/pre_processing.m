function [fileOut] = pre_processing(dataset,overwrite)
    if nargin < 2
        overwrite = -1;
    end
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

    %CARREGA DATASET
    load(['data/' dataset '.mat']);

    %NORMALIZACAO DOS DADOS
    typeNormalization = 'zscore';
    %typeNormalization = 'min-max';

    [X, mean_val, std_val,min_val, max_val] = normalization( X, typeNormalization );

    %TRATANDO SAIDA DE PROBLEMAS MULTICLASS
    classes = unique(Y);
    if length(classes)>2
      [Y] = multiclassY(Y);
    end

    %CROSS-VALIDATION
    %typeCrossValidation = 'holdout';
    typeCrossValidation = 'kfold';

    %informacoes sobre o pre-processamento
    preProcInfo = struct('dataset',dataset, ...
        'classes',classes,...
        'crossValidation',typeCrossValidation,'k',[], ...
        'normalization',typeNormalization, ...
        'mean_val', mean_val, ...
        'std_val',std_val,...
        'min_val',min_val,'max_val',max_val);

    if strcmp(typeCrossValidation,'holdout')

      % HOLDOUT  
      k = 0.7; %porcentagem de dados de treinamento
      [ Xtr, Ytr, Xtest, Ytest ] = holdout( X, Y, k ); %k-fold cross-validation
      [ Xtr, Ytr, Xval, Yval ] = holdout( Xtr, Ytr, 0.66 );

      preProcInfo.k = k;

      fileOut = ['../data/' dataset '_holdout']; %define o nome do arq de saida
        
      if exist(fileOut, 'file') %verifica se o arquivo ja existe
        if overwrite == -1
            choice = questdlg('The file allready exist, do you want to overwrite it?', 'Warning', 'Yes','No','No');
            if strcmp(choice,'No')
                overwrite = 0;
            else
                overwrite = 1;
            end
        end
        if overwrite == 0
            return; 
        end  
      end
      
      save(fileOut,'Xtr','Ytr','Xval','Yval','Xtest','Ytest','preProcInfo'); %salva arq

    elseif strcmp(typeCrossValidation,'kfold')

      % K-FOLD
      k=5; %numero de folds
      K =5;
      [ XtrFolds, YtrFolds ] = kfoldCV( X, Y, K );

      preProcInfo.k = K;

      fileOut = ['data/' dataset '_kfold' num2str(k)]; %define o nome do arq de saida

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
      
      if exist([fileOut '.mat'], 'file') && overwrite == 0%verifica se o arquivo ja existe
        if overwrite == -1
            choice = questdlg('The file already exist, do you want to overwrite it?', 'Warning', 'Yes','No','No');
            if strcmp(choice,'No')
                overwrite = 0;
            else
                overwrite = 1;
            end
        end          
        if overwrite == 0
            return; 
        end
      end
      save(fileOut,'XtrFolds','YtrFolds','XvalFolds','YvalFolds','preProcInfo'); %salva arq
    end

    preProcInfo
end
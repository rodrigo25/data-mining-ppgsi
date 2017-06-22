load('data/data_spambase.mat')

%NORMALIZACAO DOS DADOS
[X, mean_val, std_val] = normalization( X, 'zscore' ); % z-score
%[X, ~, ~, min_val, max_val] = normalization( X, 'minmax' ); % min-max

%CRIACAO DOS K-FOLDS
k=10;
[ X_folds, Y_folds ] = kfoldCV( X, Y, k );   % k-fold cross-validation

for it=1:k
  
  %SEPARACAO DOS CONJ DE TREINAMENTO E TESTE
  % Define Conj de Teste Xt e Yt
  Xt = X_folds{it}; %Pega soh os dados do fold {it}
  Yt = Y_folds{it}; 
  
  % Define Conj de treinamento Xtr e Ytr
  Xtr = X_folds; %Pega os dados de todos os folds
  Ytr = Y_folds;
  Xtr{it} = [];  %Remove os dados do fold {it}
  Ytr{it} = [];
  Xtr = cell2mat(Xtr(:)); %Transforma valores de celula para matriz
  Ytr = cell2mat(Ytr(:));
    
  
  %TREINA CLASSIFICADOR
  [ A, B ] = MLPtreina( Xtr, Ytr, [], [], hidLayers );
  
  %TESTA CLASSIFICADOR
  [ Y ] = MLPsaida( Xt, A, B );

  %DEFINE LIMIAR DE DECISAO
  Y(Y>=.5) = 1;
  Y(Y<.5) = 0;
  
  %CALCULA RESULTADOS PARCIAIS
  
end


%CALCULA RESULTADOS GERAL DO CROSS-VALIDATION


% Matriz de confusao

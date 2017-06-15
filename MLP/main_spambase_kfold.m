load('data_spambase.mat')
[N, m] = size(X);
k=10;
hidLayers = 5;

%NORMALIZACAO DOS DADOS
% z-score normalization
X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1); 

% min-max normalization
%min_val = min(X)-.1;
%max_val = max(X)+.5;
%data = zeros(N,m);
%for i=1:m
%  data(:,i) = (X(:,i)-min_val(i))/(max_val(i)-min_val(i));
%end


%CRIACAO DOS K-FOLDS
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
  [ A, B ] = MLP( Xtr, Ytr, [], [], hidLayers );
  
  %TESTA CLASSIFICADOR
  [ Y ] = saidaMLP( Xt, A, B )

  %DEFINE LIMIAR DE DECISAO
  Y(Y>=.5) = 1;
  Y(Y<.5) = 0;
  
  %CALCULA RESULTADOS PARCIAIS
  
end


%CALCULA RESULTADOS GERAL DO CROSS-VALIDATION

% acuracia
acc =  (N - sum(abs(Y - Yt)))/N

% Matriz de confusao
disp('Matriz de Confusão Binária')
confMatrix = confusionmat(Ytr,Y);
disp(array2table(confMatrix,'VariableNames',{'P','N'},'RowNames',{'P','N'}))
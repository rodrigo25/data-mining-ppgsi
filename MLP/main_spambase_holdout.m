load('data/data_spambase.mat')

%NORMALIZACAO DOS DADOS
[X, mean_val, std_val] = normalization( X, 'zscore' ); % z-score
%[X, ~, ~, min_val, max_val] = normalization( X, 'minmax' ); % min-max

[Y, classes] = multiclassY(Y);

%SEPARACAO CONJ DE TREINAMENTO E TESTE
%Xtr = X; Ytr = Y; Xt = X; Yt = Y; % trainda e testa com todos os dados
[ Xtr, Ytr, Xt, Yt ] = holdout( X, Y, 0.7 ); % holdout cross-validation
[ Xtr, Ytr, Xval, Yval ] = holdout( Xtr, Ytr, 0.66 ); % holdout cross-validation

%TREINA CLASSIFICADOR
[ A, B, MSEtrain, MSEval, epochsExecuted ] = MLPtreina2( Xtr, Ytr, Xval, Yval, 2, 10, 500, .3, 2, 1 );

epochsExecuted

%TESTA CLASSIFICADOR
Yh = MLPsaida2( Xt, 2, A, B );
[~,Yh] = max(Yh,[],2);
[~,Yt] = max(Yt,[],2);

%CALCULA RESULTADOS
multiclassConfusionMatrix( Yt, Yh, classes, 1 );

% SALVA RESULTADOS


%{
% IMPRIME RESULTADOS
disp('Matriz de Confusão Binária')
disp(array2table(confMatrix,'VariableNames',{'P','N'},'RowNames',{'P','N'}))

disp('Estatísticas')
disp(table(res(:,3),'VariableNames',{'Estatisticas'},'RowNames',res(:,2)));
%}
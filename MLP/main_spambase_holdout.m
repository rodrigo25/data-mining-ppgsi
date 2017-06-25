load('data/data_spambase.mat')

%NORMALIZACAO DOS DADOS
[X, mean_val, std_val] = normalization( X, 'zscore' ); % z-score
%[X, ~, ~, min_val, max_val] = normalization( X, 'minmax' ); % min-max

%SEPARACAO CONJ DE TREINAMENTO E TESTE
%Xtr = X; Ytr = Y; Xt = X; Yt = Y; % trainda e testa com todos os dados
[ Xtr, Ytr, Xt, Yt ] = holdout( X, Y, 0.7 ); % holdout cross-validation
[ Xtr, Ytr, Xval, Yval ] = holdout( Xtr, Ytr, 0.66 ); % holdout cross-validation

%TREINA CLASSIFICADOR
[ A, B, MSEtrain, MSEval, epochsExecuted ] = MLPtreina( Xtr, Ytr, Xval, Yval, 5, 1000, .01, 2, 1 );

epochsExecuted

%TESTA CLASSIFICADOR
Y = MLPsaida( Xt, A, B );

%DEFINE LIMIAR DE DECISAO
Y(Y>=.5) = 1;
Y(Y<.5) = 0;

%CALCULA RESULTADOS
[ confMatrix, res, res_struct] = calc_result( Yt, Y );

% SALVA RESULTADOS


% IMPRIME RESULTADOS
disp('Matriz de Confus�o Bin�ria')
disp(array2table(confMatrix,'VariableNames',{'P','N'},'RowNames',{'P','N'}))

disp('Estat�sticas')
disp(table(res(:,3),'VariableNames',{'Estatisticas'},'RowNames',res(:,2)));
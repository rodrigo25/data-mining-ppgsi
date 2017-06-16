load('data_spambase.mat')

%NORMALIZACAO DOS DADOS
[X, mean_val, std_val] = normalization( X, 'zscore' ); % z-score
%[X, ~, ~, min_val, max_val] = normalization( X, 'minmax' ); % min-max

%SEPARACAO CONJ DE TREINAMENTO E TESTE
%Xtr = X; Ytr = Y; Xt = X; Yt = Y; % trainda e testa com todos os dados
[ Xtr, Ytr, Xt, Yt ] = holdout( X, Y, 0.7 ); % holdout cross-validation

%TREINA CLASSIFICADOR
[ A, B ] = MLPtreina( Xtr, Ytr, [], [], 5 );

%TESTA CLASSIFICADOR
[ Y ] = MLPsaida( Xt, A, B );

%DEFINE LIMIAR DE DECISAO
Y(Y>=.5) = 1;
Y(Y<.5) = 0;

%CALCULA RESULTADOS
[ confMatrix, res, res_struct] = calc_result( Yt, Y );

% SALVA RESULTADOS


% IMPRIME RESULTADOS
disp('Matriz de Confusão Binária')
disp(array2table(confMatrix,'VariableNames',{'P','N'},'RowNames',{'P','N'}))

disp('Estatísticas')
disp(table(res(:,3),'VariableNames',{'Estatisticas'},'RowNames',res(:,2)));
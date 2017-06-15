load('data_winequality-red.mat');
%load('data_winequality-white.mat');

N = size(X,1);

% Z-SCORE NORALIZATION
X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1); 

% SEPARACAO CONJ DE TREINAMENTO E TESTE
Xtr = X; % teste com todos os dados
Ytr = Y;
%[ Xtr, Ytr, Xt, Yt ] = holdout( X, Y, 0.7 ); % holdout cross-validation
%[ X_folds, Y_folds ] = kfoldCV( X, Y, k );   % k-fold cross-validation


return
% TREINA MLP
[ A, B ] = MLP( Xtr, Ytr, [], [], 5 );

% CALCULA SAIDA
[ Y ] = saidaMLP( Xtr, A, B )

% SEPARA CLASSES
Y(Y>=.5) = 1;
Y(Y<.5) = 0;

% CALCULA ACURÁCIA



% MATRIZ DE CONFUSÃO
disp('Matriz de Confusão')
[confMatrix, order] = confusionmat(Ytr,Y);
order = arrayfun(@num2str, order, 'unif', 0);
order = strcat('c',order);
disp(array2table(confMatrix,'VariableNames',order,'RowNames',order))
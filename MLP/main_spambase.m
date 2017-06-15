load('data_spambase.mat')

N = size(X,1);

% Z-SCORE NORALIZATION
X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1); 

% SEPARACAO CONJ DE TREINAMENTO E TESTE
Xtr = X; % teste com todos os dados
Ytr = Y;
%[ Xtr, Ytr, Xt, Yt ] = holdout( X, Y, 0.7 ); % holdout cross-validation
%[ X_folds, Y_folds ] = kfoldCV( X, Y, k );   % k-fold cross-validation


[ A, B ] = MLP( Xtr, Ytr, [], [], 5 );

[ Y ] = saidaMLP( Xtr, A, B )

Y(Y>=.5) = 1;
Y(Y<.5) = 0;

acc =  (N - sum(abs(Y - Ytr)))/N


disp('Matriz de Confusão Binária')
confMatrix = confusionmat(Ytr,Y);
disp(array2table(confMatrix,'VariableNames',{'P','N'},'RowNames',{'P','N'}))
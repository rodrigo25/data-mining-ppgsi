load('data_winequality-red.mat');
%load('data_winequality-white.mat');

N = size(X,1);

% Z-SCORE NORALIZATION
X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1); 

[Yd, classes] = multiclassY(Y);

% SEPARACAO CONJ DE TREINAMENTO E TESTE
%Xtr = X; % teste com todos os dados
%Ytr = Y;

[ Xtr, Ytr, Xt, Yt ] = holdout( X, Yd, 0.7 ); % holdout cross-validation
%[ X_folds, Y_folds ] = kfoldCV( X, Yd, k );   % k-fold cross-validation


% TREINA MLP
[ A, B ] = MLPtreina( Xtr, Ytr, [], [], 5 );


% CALCULA SAIDA
Y = MLPsaida( Xt, A, B );

[~,Y] = max(Y,[],2);
[~,Yt] = max(Yt,[],2);

multiclassConfusionMatrix( Yt, Y, classes );
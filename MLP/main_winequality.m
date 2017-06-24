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

[~,Ytc] = max(Yt,[],2);

params = {
    20 500; 
    20 1000; 
    25 500; 
    25 1000; 
    30 500;
    30 1000;
    35 500;
    35 1000;
    40 500;
    40 1000;
    45 500;
    45 1000;
    };
accuracies = zeros(size(params,1),1);

for i=1:size(params,1)
    h = params{i,1};
    epochs = params{i,2};
    fprintf('h = %d epochs = %d\n', h, epochs);
    
    % TREINA MLP
    [ A, B ] = MLPtreina( Xtr, Ytr, [], [], h, epochs );

    % CALCULA SAIDA
    Y = MLPsaida( Xt, A, B );

    [~,Yc] = max(Y,[],2);

    [acc, ~ ] = multiclassConfusionMatrix( Ytc, Yc, classes, 2 );
    accuracies(i) = acc;    
end

accuracies
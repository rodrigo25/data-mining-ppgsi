% Gera um conjunto de dados de 4 classes e 100 instancias

N = 100;

X = zeros(N,2);
Yd = zeros(N,1);

centers = [1 1;1 5;5 1;5 5];

for i=1:4
    ini = (i-1)*25;
    count = ini + 25;
    for j=ini+1:count
        d = rand;
        if rand >= 0.5
           d = -d; 
        end
        X(j, 1) = centers(i,1)+d;
        d = rand;
        if rand >= 0.5
           d = -d; 
        end
        X(j, 2) = centers(i,2)+d;
    end
    Yd(ini+1:count, 1) = i;
end

scatter(X(:,1),X(:,2));

[Yd, classes] = multiclassY(Yd);

[ Xtr, Ytr, Xt, Yt ] = holdout( X, Yd, 0.7 ); % holdout cross-validation
%[ X_folds, Y_folds ] = kfoldCV( X, Y, k );   % k-fold cross-validation


% TREINA MLP
[ A, B ] = MLPtreina( Xtr, Ytr, [], [], 5, 5000 );

% CALCULA SAIDA
Y = MLPsaida( Xt, A, B );

[~,Y] = max(Y,[],2);
[~,Yt] = max(Yt,[],2);

multiclassConfusionMatrix( Yt, Y, classes );
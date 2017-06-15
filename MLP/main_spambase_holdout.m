load('data_spambase.mat')
[N, m] = size(X);

%NORMALIZACAO DOS DADOS
%z-score normalization
X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1); 

% min-max normalization
%min_val = min(X)-.1;
%max_val = max(X)+.5;
%data = zeros(N,m);
%for i=1:m
%  data(:,i) = (X(:,i)-min_val(i))/(max_val(i)-min_val(i));
%end


%SEPARACAO CONJ DE TREINAMENTO E TESTE
%Xtr = X; Ytr = Y; Xt = X; Yt = Y; % teste com todos os dados
[ Xtr, Ytr, Xt, Yt ] = holdout( X, Y, 0.7 ); % holdout cross-validation


%TREINA CLASSIFICADOR
[ A, B ] = MLP( Xtr, Ytr, [], [], 5 );

%TESTA CLASSIFICADOR
[ Y ] = saidaMLP( Xt, A, B )

%DEFINE LIMIAR DE DECISAO
Y(Y>=.5) = 1;
Y(Y<.5) = 0;

%CALCULA RESULTADOS

% acuracia
acc =  (N - sum(abs(Y - Yt)))/N

% Matriz de confusao
disp('Matriz de Confusão Binária')
confMatrix = confusionmat(Ytr,Y);
disp(array2table(confMatrix,'VariableNames',{'P','N'},'RowNames',{'P','N'}))
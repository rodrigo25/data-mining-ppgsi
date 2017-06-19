load('data/data_test_gate.mat');
%load('data/data_test_1.mat')
%load('data/data_spambase.mat')
Xtr = X;
Ytr = Y_XOR;

N = size(Xtr,1);

[ A, B ] = MLPtreina( Xtr, Ytr, [], [], 10 );

[ Y ] = MLPsaida( Xtr, A, B )

Y(Y>=.5) = 1;
Y(Y<.5) = 0;



[ confMatrix, res, res_struct] = calc_result( Ytr, Y );

% IMPRIME RESULTADOS
disp('Matriz de Confusão Binária')
disp(array2table(confMatrix,'VariableNames',{'P','N'},'RowNames',{'P','N'}))

disp('Estatísticas')
disp(table(res(:,3),'VariableNames',{'Estatisticas'},'RowNames',res(:,2)));



%scatter(X((Y_AND==1),1),X((Y_AND==1),2),'x');
%hold on
%scatter(X((Y_AND==0),1),X((Y_AND==0),2),'o');
%load('data_test_gate.mat');
load('data_test_1.mat')
%load('spambase.mat')
%Xtr = X;
%Ytr = Y_XOR;

N = size(Xtr,1);

[ A, B ] = MLP( Xtr, Ytr, [], [], 10 );

[ Y ] = saidaMLP( Xtr, A, B )

Y(Y>=.5) = 1;
Y(Y<.5) = 0;

acc =  (N - sum(abs(Y - Ytr)))/N
precision = sum(Y(Ytr==1)==1)

disp('Matriz de Confusão Binária')
[confMatrix,order] = confusionmat(Ytr,Y);
disp(array2table(confMatrix,'VariableNames',{'P','N'},'RowNames',{'P','N'}))


%scatter(X((Y_AND==1),1),X((Y_AND==1),2),'x');
%hold on
%scatter(X((Y_AND==0),1),X((Y_AND==0),2),'o');
load('data_test_gate.mat');
%load('data_test_1.mat')
%load('spambase.mat')
Xtr = X;
Ytr = Y_XOR;

N = size(Xtr,1);

[ A, B ] = MLPtreina( Xtr, Ytr, [], [], 10 );

[ Y ] = MLPsaida( Xtr, A, B )

Y(Y>=.5) = 1;
Y(Y<.5) = 0;



[confMatrix] = confusionmat(Ytr,Y,'order',[1,0]);

TP = confMatrix(1,1);
FN = confMatrix(1,2);
FP = confMatrix(2,1);
TN = confMatrix(2,2);

TPR = TP/(TP+FN);
TNR = TN/(TN+FP);
PPV = TP/(TP+FP)
NPV = TN/(TN+FN);
FNR = FN/(FN+TP);
FPR = FP/(FP+TN);
FDR = FP/(FP+TP);
FOR = FN/(FN/TN);
ACC = (TP+TN)/(TP+TN+FP+FN)
Fscore = 2*(PPV*TPR)/(PPV+TPR);

disp('Matriz de Confusão Binária')
disp(array2table(confMatrix,'VariableNames',{'P','N'},'RowNames',{'P','N'}))

disp(table([TPR;TNR;PPV;FNR;FPR],'RowNames',{'Recall' 'Specificity' 'Precision' 'FNR' 'FPR'}));

%scatter(X((Y_AND==1),1),X((Y_AND==1),2),'x');
%hold on
%scatter(X((Y_AND==0),1),X((Y_AND==0),2),'o');
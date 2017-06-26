%CARREGA DATASET
load('../data/data_spambase_kfold5.mat');
  
maxIt = 1000;
neur = 50;

Ymlp = [];
Yd = [];
  
for k=1:5
  fprintf(['\n## Fold ' num2str(k) '\n']);
  [Xtr, Ytr, Xtest, Ytest, Xval, Yval] = createDatasets(XtrFolds, YtrFolds, XvalFolds, YvalFolds, k);
  Yd = [Yd; Ytest];
    
  %TREINA CLASSIFICADOR
  [ A, B, ERROtr, ERROval, ALFA ] = MLP_alfaAdaptativo( Xtr, Ytr, Xval, Yval, neur, maxIt,1);
  redeConfig.Alfa = ['Adaptativo'];
  
  %TESTA CLASSIFICADOR
  Ymlp = [Ymlp; MLPsaida( Xtest, A, B )];
  
  figure
  plot(ERROtr,'DisplayName','Erro de Treinamento')
  hold on
  plot(ERROval,'DisplayName','Erro de Validacao')
  legend('show')
  title(['Erro Quadrático Médio - fold ' num2str(k)])
end

Y(Ymlp>=.5) = 1;
Y(Ymlp<.5) = 0;

[ confMatrix, res, res_struct] = calc_result( Yd, Y );

% IMPRIME RESULTADOS
disp('Matriz de Confusão Binária')
disp(array2table(confMatrix,'VariableNames',{'P','N'},'RowNames',{'P','N'}))

disp('Estatísticas')
disp(table(res(:,3),'VariableNames',{'Estatisticas'},'RowNames',res(:,2)));



%scatter(X((Y_AND==1),1),X((Y_AND==1),2),'x');
%hold on
%scatter(X((Y_AND==0),1),X((Y_AND==0),2),'o');
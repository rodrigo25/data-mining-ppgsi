function [TPR, FPR] = spambase_kfoldCV(classificador, neur, dirPai)
  
  %CRIA O DIRETORIO DE RESULTADOS
  dirName = [dirPai classificador num2str(neur) 'h/'];
  if ~exist(dirName, 'dir') % se o dir nao existe, ele eh criado
    mkdir(dirName);
  end
  
  %CARREGA DATASET
  load('../data/data_spambase_kfold5.mat');
  
  maxIt = 1000;
 
  redeConfig = struct('Nome',classificador,'Otimizacao','Gradiente','Alfa',[],'MaxIt',maxIt,'Neuronios',neur,'Camadas',1,'Limiar',[]);

  Ymlp = [];
  Yd = [];
  
  for k=1:5
    fprintf(['\n## Fold ' num2str(k) '\n']);
    [Xtr, Ytr, Xtest, Ytest, Xval, Yval] = createDatasets(XtrFolds, YtrFolds, XvalFolds, YvalFolds, k);
    Yd = [Yd; Ytest];
    
    %TREINA CLASSIFICADOR
    switch classificador
      case 'MLP_alfaAdaptativo'
        [ A, B ] = MLP_alfaAdaptativo( Xtr, Ytr, Xval, Yval, neur, maxIt,0);
        redeConfig.Alfa = ['Adaptativo'];
      case 'MLP_clodoaldo'
        [ A, B ] = MLP_clodoaldo( Xtr, Ytr, neur, maxIt );
        redeConfig.Alfa = ['Bisseção'];
      otherwise
        error('Rede Inexistente')      
    end
      
    %TESTA CLASSIFICADOR
    Ymlp = [Ymlp; MLPsaida( Xtest, A, B )];
    
    %SALVA DADOS
    save([dirName 'Rede_' classificador num2str(neur) 'h_k' num2str(k)],'A','B','Ymlp');
    
  end
  
  
  TPR = [];
  FPR = [];
  fN = 0;
  
  %DEFINE LIMIAR DE DECISAO
  for threshold=0:0.005:1
    fN = fN +1;
    
    Y(Ymlp>=threshold) = 1;
    Y(Ymlp<threshold) = 0;

    redeConfig.Limiar = threshold;
    
    %CALCULA RESULTADOS
    [ confMatrix, ~, res_struct] = calc_result( Yd, Y );

    % SALVA RESULTADOS
    save([dirName 'Resultados' num2str(fN)],'confMatrix','res_struct','threshold');
    create_log(res_struct, dirName, redeConfig, fN);
    
    %ARMAZENA TPR E FPR PARA MONTAR A CURVA ROC
    TPR = [TPR; res_struct(2).TPR];
    FPR = [FPR; res_struct(2).FPR];
    
  end
  
  
end
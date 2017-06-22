function [ ] = createLog( dataset, sizeX, typeNorm,  dim, alfaIni, raioIni, lambda, tau, fDist, itMax, TimeTrain, QuantErrorHist, TopolErrorHist, dirName )

  file_name = [dirName 'log.txt'];
  fileID = fopen(file_name, 'w');
  fprintf(fileID, '\n\t Log de Treinamento da Rede SOM\n\n');
  
  fprintf(fileID, ' ---------------------------------------\n');
  fprintf(fileID, '              Base de Dados             \n');
  fprintf(fileID, ' ---------------------------------------\n');
  fprintf(fileID, ' Nome: \t\t\t\t\t\t%s\n', dataset);
  fprintf(fileID, ' Quantidade de Instâncias: \t%d\n', sizeX(1));
  fprintf(fileID, ' Dimensionalidade: \t\t\t%d\n', sizeX(2));
  fprintf(fileID, ' Tipo de Normalização: \t\t%s\n', typeNorm);
  
  
  fprintf(fileID, '\n\n ---------------------------------------\n');
  fprintf(fileID, '         Parametrização da Rede         \n');
  fprintf(fileID, ' ---------------------------------------\n');
  fprintf(fileID, ' Dimensionalidade da saída: %dx%d\n', dim);
  fprintf(fileID, ' Alfa Inicial: \t\t\t\t%3.3f\n', alfaIni);
  fprintf(fileID, ' Raio Inicial: \t\t\t\t%d\n', raioIni);
  fprintf(fileID, ' Lambda: \t\t\t\t\t%d\n', lambda);
  fprintf(fileID, ' Tau: \t\t\t\t\t\t%d\n', tau);
  if strcmp(fDist,'e')
    fprintf(fileID, ' Função de Distância: \t\t%s\n', 'Euclidiana');
  elseif strcmp(fDist,'m')
    fprintf(fileID, ' Função de Distância: \t\t\t%s\n', 'Manhattan');
  end
  fprintf(fileID, ' Máximo de Iterações: \t  %5d\n', itMax);
  fprintf(fileID, ' Topologia: \t\t\t\t\tRetangular\n');
  fprintf(fileID, ' Função de Vizinhança: \t\tGaussiana\n');
   
  
  fprintf(fileID, '\n\n ---------------------------------------\n');
  fprintf(fileID, '               Resultados              \n');
  fprintf(fileID, ' ---------------------------------------\n');
  fprintf(fileID, ' Tempo de treinamento (s): \t%7.7f\n', TimeTrain);
  fprintf(fileID, ' Erro de Quatização: \t\t%7.7f\n',QuantErrorHist);
  fprintf(fileID, ' Erro Topológico: \t\t\t%7.7f\n', TopolErrorHist);
  

  fclose(fileID);


end


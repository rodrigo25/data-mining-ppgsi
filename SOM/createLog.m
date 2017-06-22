function [ ] = createLog( dataset, sizeX, typeNorm,  dim, alfaIni, raioIni, lambda, tau, fDist, itMax, TimeTrain, QuantErrorHist, TopolErrorHist, dirName )

  file_name = [dirName 'log.txt'];
  fileID = fopen(file_name, 'w');
  fprintf(fileID, '\n\t Log de Treinamento da Rede SOM\n\n');
  
  fprintf(fileID, ' ---------------------------------------\n');
  fprintf(fileID, '              Base de Dados             \n');
  fprintf(fileID, ' ---------------------------------------\n');
  fprintf(fileID, ' Nome: \t\t\t\t\t\t%s\n', dataset);
  fprintf(fileID, ' Quantidade de Inst�ncias: \t%d\n', sizeX(1));
  fprintf(fileID, ' Dimensionalidade: \t\t\t%d\n', sizeX(2));
  fprintf(fileID, ' Tipo de Normaliza��o: \t\t%s\n', typeNorm);
  
  
  fprintf(fileID, '\n\n ---------------------------------------\n');
  fprintf(fileID, '         Parametriza��o da Rede         \n');
  fprintf(fileID, ' ---------------------------------------\n');
  fprintf(fileID, ' Dimensionalidade da sa�da: %dx%d\n', dim);
  fprintf(fileID, ' Alfa Inicial: \t\t\t\t%3.3f\n', alfaIni);
  fprintf(fileID, ' Raio Inicial: \t\t\t\t%d\n', raioIni);
  fprintf(fileID, ' Lambda: \t\t\t\t\t%d\n', lambda);
  fprintf(fileID, ' Tau: \t\t\t\t\t\t%d\n', tau);
  if strcmp(fDist,'e')
    fprintf(fileID, ' Fun��o de Dist�ncia: \t\t%s\n', 'Euclidiana');
  elseif strcmp(fDist,'m')
    fprintf(fileID, ' Fun��o de Dist�ncia: \t\t\t%s\n', 'Manhattan');
  end
  fprintf(fileID, ' M�ximo de Itera��es: \t  %5d\n', itMax);
  fprintf(fileID, ' Topologia: \t\t\t\t\tRetangular\n');
  fprintf(fileID, ' Fun��o de Vizinhan�a: \t\tGaussiana\n');
   
  
  fprintf(fileID, '\n\n ---------------------------------------\n');
  fprintf(fileID, '               Resultados              \n');
  fprintf(fileID, ' ---------------------------------------\n');
  fprintf(fileID, ' Tempo de treinamento (s): \t%7.7f\n', TimeTrain);
  fprintf(fileID, ' Erro de Quatiza��o: \t\t%7.7f\n',QuantErrorHist);
  fprintf(fileID, ' Erro Topol�gico: \t\t\t%7.7f\n', TopolErrorHist);
  

  fclose(fileID);


end


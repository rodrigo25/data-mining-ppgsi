function [ ] = plotDeltas(DeltaMeanHist, DeltaMaxHist, dirName)

  im_deltaMean = figure;
  plot(DeltaMeanHist)
  title('Atualizacao Média dos Pesos (Delta Médio)')
  print(im_deltaMean,[dirName 'deltaMean'],'-dpng');
  
  im_deltaMax = figure;
  plot(DeltaMaxHist)
  title('Atualizacao Máxima dos Pesos (Delta Máximo)')
  print(im_deltaMax,[dirName 'deltaMax'],'-dpng');

end


function [ ] = plotDeltas(DeltaMeanHist, DeltaMaxHist, dirName)

  im_deltaMean = figure;
  plot(DeltaMeanHist)
  title('Atualizacao M�dia dos Pesos (Delta M�dio)')
  print(im_deltaMean,[dirName 'deltaMean'],'-dpng');
  
  im_deltaMax = figure;
  plot(DeltaMaxHist)
  title('Atualizacao M�xima dos Pesos (Delta M�ximo)')
  print(im_deltaMax,[dirName 'deltaMax'],'-dpng');

end


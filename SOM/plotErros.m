function [ ] = plotErros(QuantErrorHist, TopolErrorHist, dirName)

  im_quantError = figure;
  plot(QuantErrorHist)
  title('Erro de Quantização')
  print(im_quantError,[dirName 'erroQuantizacao'],'-dpng');
  
  im_topolError = figure;
  plot(TopolErrorHist)
  title('Erro Topológico')
  print(im_topolError,[dirName 'erroTopologico'],'-dpng');

end


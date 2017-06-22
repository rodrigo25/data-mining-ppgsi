function [ ] = plotErros(QuantErrorHist, TopolErrorHist, dirName)

  im_quantError = figure;
  plot(QuantErrorHist)
  title('Erro de Quantiza��o')
  print(im_quantError,[dirName 'erroQuantizacao'],'-dpng');
  
  im_topolError = figure;
  plot(TopolErrorHist)
  title('Erro Topol�gico')
  print(im_topolError,[dirName 'erroTopologico'],'-dpng');

end


function [ ] = plotInfluencia( X, W, H, BMU, inflMin, dirName)

  im_influencia = figure;
  scatter(W(H>=inflMin,1), W(H>=inflMin,2),'k.');
  hold on
  
  for i=1:size(X,1)
    if H(BMU(i))> inflMin
      plot([W(BMU(i),1) X(i,1)], [W(BMU(i),2) X(i,2)],'r'); 
      %viscircles([tracos{i,2}(1,1) tracos{i,3}(1,1)], tracos{i,4});
    end
  end

  title(['Influência dos Neurônios com infMin ' num2str(inflMin)])
  print(im_influencia,[dirName 'influencia(min' num2str(inflMin) ')'],'-dpng');
end

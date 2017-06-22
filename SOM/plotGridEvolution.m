function [  ] = plotGridEvolution( WHist, dimGrid, dirName )

  %Ns - Qtd Neuronios totais
  %DimW - Dimensao do espaco de entrada
  [Ns, dimW] = size(WHist{1});  

  im_gridEvol = figure;
  
  for iW=1:4

    W = WHist{iW,1};
    subplot(2,2,iW);
    
    switch dimW
  
    %Caso o grid seja de dimensão 2
    case 2
      scatter(W(:,1), W(:,2)) %Plota os pontos do grid
      if iW==0
        title( 'Grid iniciado aleatóriamente');
      else
        title( ['Grid após ' num2str(WHist{iW,2}) ' iterações']);
      end
      hold on
      for i=1:Ns %Para cada ponto no Grid
        if (rem(i-1,dimGrid(2)) ~= dimGrid(2)-1)  %Verifica se não está na borda direita
          plot([W(i,1) W(i+1,1)],[W(i,2) W(i+1,2)],'r') %Desenha linha pra direita
        end
        if (fix((i-1)/dimGrid(2)) ~= dimGrid(1)-1)  %Verifica se não está na borda inferior
          plot([W(i,1) W(i+dimGrid(2),1)],[W(i,2) W(i+dimGrid(2),2)],'r') %Desenha linha pra baixo
        end
      end
        
      %Caso o grid seja de dimensão 3
    case 3
      scatter3(W(:,1), W(:,2), W(:,3)) %Plota os pontos do grid
      if iW==1
        title( 'Grid iniciado aleatóriamente');
      else
        title( ['Grid após ' num2str(WHist{iW,2}) ' iterações']);
      end
      hold on
      for i=1:Ns %Para cada ponto no Grid
        if (rem(i-1,dimGrid(2)) ~= dimGrid(2)-1)  %Verifica se não está na borda direita
          plot3([W(i,1) W(i+1,1)],[W(i,2) W(i+1,2)],[W(i,3) W(i+1,3)],'r') %Desenha linha pra direita
        end
        if (fix((i-1)/dimGrid(2)) ~= dimGrid(1)-1)  %Verifica se não está na borda inferior
          plot3([W(i,1) W(i+dimGrid(2),1)],[W(i,2) W(i+dimGrid(2),2)],[W(i,3) W(i+dimGrid(2),3)],'r') %Desenha linha pra baixo
        end
      end
    
    otherwise
      warning('Unexpected plot type. No plot created.');
    
    end %case
    
  end %for
  
  print(im_gridEvol,[dirName 'gridEvolution'],'-dpng');
  %savefig(im_grid,[dirName 'grid']);

end


function [  ] = plotGrid( W, dimGrid, dirName )

%Ns - Qtd Neuronios totais
%DimW - Dimensao do espaco de entrada
[Ns, dimW] = size(W);

switch dimW

  %Caso o grid seja de dimens�o 2
  case 2
    im_grid = figure; 
    scatter(W(:,1), W(:,2)) %Plota os pontos do grid
    hold on
    for i=1:Ns %Para cada ponto no Grid
      if (rem(i-1,dimGrid(2)) ~= dimGrid(2)-1)  %Verifica se n�o est� na borda direita
        plot([W(i,1) W(i+1,1)],[W(i,2) W(i+1,2)],'r') %Desenha linha pra direita
      end
      if (fix((i-1)/dimGrid(2)) ~= dimGrid(1)-1)  %Verifica se n�o est� na borda inferior
        plot([W(i,1) W(i+dimGrid(2),1)],[W(i,2) W(i+dimGrid(2),2)],'r') %Desenha linha pra baixo
      end
    end
    title('Grid dos Neur�nios de Sa�da');
    print(im_grid,[dirName 'grid'],'-dpng');
    %savefig(im_grid,[dirName 'grid']);
    
    
    
    
  %Caso o grid seja de dimens�o 3
  case 3
    im_grid = figure; 
    scatter3(W(:,1), W(:,2), W(:,3)) %Plota os pontos do grid
    hold on
    for i=1:Ns %Para cada ponto no Grid
      if (rem(i-1,dimGrid(2)) ~= dimGrid(2)-1)  %Verifica se n�o est� na borda direita
        plot3([W(i,1) W(i+1,1)],[W(i,2) W(i+1,2)],[W(i,3) W(i+1,3)],'r') %Desenha linha pra direita
      end
      if (fix((i-1)/dimGrid(2)) ~= dimGrid(1)-1)  %Verifica se n�o est� na borda inferior
        plot3([W(i,1) W(i+dimGrid(2),1)],[W(i,2) W(i+dimGrid(2),2)],[W(i,3) W(i+dimGrid(2),3)],'r') %Desenha linha pra baixo
      end
    end
    title('Grid dos Neur�nios de Sa�da');
    print(im_grid,[dirName 'grid'],'-dpng');
    %savefig(im_grid,[dirName 'grid']);
  
    
    
    
  otherwise
    warning('Unexpected plot type. No plot created.');

end


function [ output_args ] = plotGrid( W, Ns, Nx, dirName )

dimW = size(W,2); %Descobre dimensão do Grid

switch dimW
   
  %Caso o grid seja de dimensão 2
  case 2
    im_grid = figure; 
    scatter(W(:,1), W(:,2)) %Plota os pontos do grid
    hold on
    for i=1:Ns %Para cada ponto no Grid
      if (rem(i-1,Nx) ~= Nx-1)  %Verifica se não está na borda direita
        plot([W(i,1) W(i+1,1)],[W(i,2) W(i+1,2)],'r') %Desenha linha pra direita
      end
      if (fix((i-1)/Nx) ~= Nx-1 && Ns ~= Nx)  %Verifica se não está na borda inferior
        plot([W(i,1) W(i+Nx,1)],[W(i,2) W(i+Nx,2)],'r') %Desenha linha pra baixo
      end
    end
    print(im_grid,[dirName 'grid'],'-dpng');
    %savefig(im_grid,[dirName 'grid']);
    
    
    
    
  %Caso o grid seja de dimensão 3
  case 3
    im_grid = figure;
    scatter3(W(:,1), W(:,2), W(:,3)) %Plota os pontos do grid
    hold on
    for i=1:Ns %Para cada ponto no Grid
      if (rem(i-1,Nx) ~= Nx-1)  %Verifica se não está na borda direita
        plot3([W(i,1) W(i+1,1)],[W(i,2) W(i+1,2)],[W(i,3) W(i+1,3)],'r') %Desenha linha pra direita
      end
      if (fix((i-1)/Nx) ~= Nx-1)  %Verifica se não está na borda inferior
        plot3([W(i,1) W(i+Nx,1)],[W(i,2) W(i+Nx,2)],[W(i,3) W(i+Nx,3)],'r') %Desenha linha pra baixo
      end
    end
    print(im_grid,[dirName 'grid'],'-dpng');
    %savefig(im_grid,[dirName 'grid']);
  
    
    
    
  otherwise
    warning('Unexpected plot type. No plot created.');

end


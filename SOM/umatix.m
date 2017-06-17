function [ output_args ] = umatix( W, Nx, Ns, dirName )



grid = (2*Nx-1);

for i=1:(2*Nx-1)
  for j=1:(2*Nx-1)
    
    if mod(i,2)~=0 %linha impar
      if mod(j,2)~=0 %coluna impar
        
        
        
      else %coluna par
        
      end
      
    else %linha par
      if mod(j,2)~=0 %coluna impar
        
      else %coluna par
      end
    end
      
  end
end






grid = (Nx);

for i=1:Ns
  t = 0;
  soma = 0;
  if((fix((i-1)/Nx) ~= 0))%linha pra cima
    t = t+1;
    soma = soma + norm(W(i)-W(i-Nx));
    %plot([W(i,1) W(i-Nx,1)],[W(i,2) W(i-Nx,2)],'r')
  end 
  if (rem(i-1,Nx) ~= Nx-1)%linha pra direita
    t = t+1;
    soma = soma + norm(W(i)-W(i+1));
    %plot([W(i,1) W(i+1,1)],[W(i,2) W(i+1,2)],'r')
  end
  if (fix((i-1)/Nx) ~= Nx-1)%linha pra baixo
    t = t+1;
    soma = soma + norm(W(i)-W(i+Nx));
    %plot([W(i,1) W(i+Nx,1)],[W(i,2) W(i+Nx,2)],'r')
  end
  if (rem(i-1,Nx) ~= 0)%linha pra esquerda
    t = t+1;
    soma = soma + norm(W(i)-W(i-1));
    %plot([W(i,1) W(i-1,1)],[W(i,2) W(i-1,2)],'r')
  end

  grid((fix((i-1)/Nx)+1), (rem(i-1,Nx)+1)) = soma/t;
end

im_grid = figure;
maximo  = max(max(grid));
minimo  = min(min(grid));
data = ((grid-minimo)/(maximo-minimo))*255;
image(data);
colorMap = jet(256);
colormap(colorMap);
colorbar;

print(im_grid,[dirName 'u-matrix'],'-dpng');

end


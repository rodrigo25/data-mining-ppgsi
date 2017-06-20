function [  ] = plotOriginalData( X, dirName )

dimX = size(X,2); %Descobre dimens�o do Grid

switch dimX
   
  %Caso os dados sejam de dimens�o 2
  case 2
    im_originalData = figure;
    scatter(X(:,1),X(:,2));
    print(im_originalData,[dirName 'originalData'],'-dpng');
    
  %Caso os dados sejam de dimens�o 3
  case 3
    im_originalData = figure;
    scatter3(X(:,1),X(:,2),X(:,3));
    print(im_originalData,[dirName 'originalData'],'-dpng');

  otherwise
    warning('Unexpected plot type. No plot created.');
    
end


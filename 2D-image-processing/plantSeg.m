clc; 
clear;
close all
fclose('all');

%camera parameters
pixelSize=5.86e-04; %cm
fx=1.4135021364891049e+03; %pixel
fc=15.24; %cm; =6 inch
scalFactor= fc/(fx*pixelSize)*pixelSize; %cm/pixel

tablerow=0;

write_to_original=0;
write_to_processed=1;

merge_plant=1;
test=0;
test_id = 'M470-2_W_66.85_4';
test_date='2021-4-16'; %
modify=0;
modify_pot_id=test_id;

fid_label=fopen('experiment_label.txt','r');

[status, msg, msgID] = mkdir('merged_2d');


id_count=0;
while ~feof(fid_label)    
    line=fgets(fid_label);
    id_count=id_count+1;
    vec=strsplit(string(line),'_');
    water_type=vec(2);
    weight=str2num(vec(3));
    pot_id=regexprep(line,'\s+',''); %replace new line
    dead_plant = 0;

    if modify
        if strcmp(modify_pot_id, pot_id)>0
            disp('find modify id')
        else
            continue;
        end
    end
    
    if write_to_original
        original_floder_str=strcat('2d_images/',pot_id,'/original');
        [status, msg, msgID] = mkdir(original_floder_str);
    end
    
    if write_to_processed
        processed_floder_str=strcat('2d_images/',pot_id,'/processed');
        [status, msg, msgID] = mkdir(processed_floder_str);
    end
    
    firstday=1;
    firstone=1;
    firstImg = 1;
    date_count=0;
    fid_date=fopen('experiment_date.txt','r');
    empty_pot=0;
    area_ref=0;
    convex_area_ref=0;
    perimeter_ref=0;
    day_ref=0;
    if test
        if strcmp(test_id, pot_id)>0
            disp('find test id')
        else
            continue;
        end
    end

    while ~feof(fid_date)  
        
        close all;
        genotype=pot_id(1:3);
        date=fgets(fid_date);
        date=regexprep(date,'\s+',''); %replace new line
        
        date_count=date_count+1;

        if test
            if strcmp(test_date, date)>0
                disp('find test date')
            else
                continue;
            end
        end
        
        str=strcat(date,'/',pot_id,'/',pot_id,'.bmp');
        
        addhue=0;
        add_exc_v=0;
  
        if (exist(str) == 0) 
            disp(strcat(str,' not exist'))
            continue;
        end
        
        tablerow=tablerow+1;
  
        a=imread(str);

        if write_to_original
            outputName=strcat(original_floder_str,'/',pot_id,'_',date,'.bmp');
            imwrite(a,outputName)
        end
        
        %detect circle, once for each pot  
        
        if firstday==1 
            [row col rad] = circlefinder(a,350,370);
            % visualize
            % a = RGBCircle(a,row(1),col(1),0.95*rad(1), [255 0 0], 4);  
        end
          
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% excessive green

        [m,n,k]=size(a);
        
        exg=zeros(m,n);

        for i=1:1200
            for j=1:1920
                if j>1500 | j<400
                    exg(i,j)=0; 
                else
                    r=double(a(i,j,1));
                    g=double(a(i,j,2));
                    b=double(a(i,j,3));
                    exg(i,j)=(2*g-r-b)/(r+g+b);
                end
            end
        end
 

        idx=exg>0.15;
        %figure; histogram(exg(idx))

        %Global histogram threshold using Otsu's method
        [counts,x] = imhist(exg(idx),16);
        %stem(x,counts)
        osTh = otsuthresh(counts);
        if osTh<0.20 | osTh >0.3
            osTh=0.2667;
        end

        osTh = 0.2;
        
        bw=imbinarize(exg,osTh); 

        [l,num]=bwlabel(bw);
        
        if(num==0)
            continue;
        end

        biggestSev = bwareafilt(bw, [500 5000000]); 

        [l,num]=bwlabel(biggestSev);
        
        if num==0 
            biggestSev=bwareafilt(bw,1,'largest');
        end
        
        temp = ~bwareaopen(~biggestSev, 500); %fill holes
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% fill holes 

        plant_region=uint8(zeros(m,n,3)*255);
  
        for i=1:m
            for j=1:n       
               if temp(i,j)==1
                    plant_region(i,j,:)=a(i,j,:);
               end
            end
        end

        green_part=imbinarize(plant_region(:,:,2),'global');
        

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% main plant
       main_plant=or(green_part,biggestSev);  
        
       if nnz(main_plant)<50 
           disp("empty pot");
           continue;
       end
        
       final = main_plant;
       
       if date_count>30 & weight < 100
           addhue=1;
       end
       
       if addhue
            hsv=rgb2hsv(a);
            hue_within_pot=zeros(m,n);
            for i=1:1200
                for j=1:1920
                    dis=sqrt(power(i-row,2)+power(j-col,2));
                    if dis<0.7*rad(1)  & int64(a(i,j,1))+int64(a(i,j,2))+int64(a(i,j,3))<400 %600
                           hue_within_pot(i,j)=hsv(i,j,1);
                    end
                end
            end

            hue_within_pot=bwareafilt(imbinarize(hue_within_pot,0.2), [500 5000000]);
            
            se = strel('square',3);
            for i=1:3
                hue_within_pot = imerode(hue_within_pot,se);
            end
            
            hue_within_pot= bwareafilt(hue_within_pot, [200 5000000]); 

            for i=1:3
                hue_within_pot = imdilate(hue_within_pot,se);
            end    
            
            convex_hue=bwconvhull(hue_within_pot);
            s_ratio=nnz(convex_hue)/(3.14*0.9.^2*rad(1).^2);

            if s_ratio>0.9  %over saturated
                rad=and(final,hue_within_pot);  
            else
                final=or(final,hue_within_pot);
                final = ~bwareaopen(~final, 500); %fill holes
            end 
         end
            

         if date_count > 20
                final= bwareafilt(final, [1000 5000000]);
         end

        if nnz(final)<100  %empty
            disp('an empty pot');
            dead_plant = 1;
            continue;
        end

        %%%%%%%%%%%%%%%%%%
        radius_ratio=1.4;
        if water_type == "W" & weight > 100
            radius_ratio = 1.4;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%5
        RGB=uint8(zeros(m,n,3)*255);
         
        for i=1:m
            for j=1:n    
                 dis=sqrt(power(i-row,2)+power(j-col,2));
                 if final(i,j)==1 & dis<radius_ratio*rad(1) %& water_type=="H"
                    RGB(i,j,:)=a(i,j,:);
                 end
            end
        end  
        
      
       if merge_plant && date_count>20 
            if firstImg
                 newImg = RGB(101:1100,411:1410,:);
                 firstImg = 0;
            else
                 newImg = cat(2,newImg,RGB(101:1100,411:1410,:));
            end 
       end
       
        if write_to_processed
           outputName=strcat(processed_floder_str,'/',pot_id,'_',date,'_plant.bmp');
           imwrite(RGB,outputName)
           disp(outputName)
        end 
            outputName=strcat('2d_images/',pot_id,'/processed/',pot_id,'.bmp');

      if test
        figure; 
        title(strcat(pot_id,date));
        subplot(2,2,1); imshow(a);
        subplot(2,2,2); imshow(exg);
        subplot(2,2,3); imshow(final);
        subplot(2,2,4); imshow(RGB);
     end
      firstday=0;
    end
    fclose(fid_date)
    
    if merge_plant & ~test 
        outputName=strcat('merged_2d/',pot_id,'.bmp');
        newImg = imresize(newImg, 0.2);
        imwrite(newImg,outputName);
        disp('write merged')
    end
        
end
    
fclose('all');
   

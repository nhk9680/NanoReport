# 20190828

## DONE
-
-

### MMDetection
- PNG RGB convert 과정을 보기 위해 convert 메소드를 열어보았으나 `ImagingCore` 클래스를 step할 수 없어 중단
- object와 class의 np.unique() 값이 [0,128,192,224]로 같음. 즉 각 index에 1:1 매핑이 아니라는 것. 또한 color 값이 아니라는 것.

- np.unique() 했을때 [0,1,15,255] 가 나오는데,

> alphabetical order (1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car , 8=cat, 9=chair, 10=cow, 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor)

- 이렇게 보았을 때 1 비행기 맞고 15 사람 맞다.

- [PNG Decoder](https://github.com/lvandeve/lodepng) 를 이용해서, 어느 부분에 class number가 매핑되는지 알아볼 예정.

### PASCAL VOC Development Kit

```
다음 사용 중 오류가 발생함: create_segmentations_from_detections (line 32)
Could not find detection results file to use to create segmentations (D:/Users/user2/Downloads/VOCdevkit/results/VOC2012/Main/comp3_det_val_aeroplane.txt not found)
```

- `drawnow` ~= `plt.show()`
- 결과파일에서 `[ids,confs,xmin,ymin,xmax,ymax]` 를 로드함  
```
t=[
    ids 
    num2cell(ones(numel(ids),1)*clsnum) 
    num2cell(confs) 
    num2cell([xmin ymin xmax ymax],2)
    ];
```
- t에 위에서 불러온 숫자를 cell형태(matrix)로 변환

> dets(n+(1:numel(ids))) = cell2struct(t,{'id' 'clsnum' 'conf' 'bbox'},2);
- ?

> % Write out the segmentations

- 밑부분은 경로 생성하는 거인듯

> % load test set

- 유력한 코드 목록
    - cmap = VOClabelcolormap(255);
    - [instim,classim]= convert_dets_to_image(imginfo.Width, imginfo.Height,vdets,confidence);
    - imwrite(instim,cmap,instlabelfile);
    - imwrite(classim,cmap,classlabelfile);    


---
### irrelevant
-
---
## TODO
- MATLAB 코드 voc셋으로 step-by-step
    - cell2struct 부터 리딩
---
.

.

.

### TMI
기상시간 7:30

출근시각 10:30

날씨: 흐림

외출시간
- 점심 (14:30~15:30)
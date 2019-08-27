# 20190827

## DONE
- kaist rgbt annotations에는 15랑 18버전이 있는데, 후자가 더 정확도가 개선된 파일이다.
### MMDetection
- PNG 포맷 문제가 아니라, `OpenCV`와 `PIL.Image`의 차이였음
- 대표적인 Python image library인 `OpenCV`, `pillow`, `matplotlib` 를 이용해 이미지를 로드 후 비교

![Figure_1](https://user-images.githubusercontent.com/30471027/63744279-d8aa2b00-c8d9-11e9-92b6-b5c20798d7bd.png)


| loaded by | shape | np.unique() |
| ------------- | ------------- | ------------- |
| original  | 281, 500, 3  | array([  0, 128, 192, 224], dtype=uint8) |
| OpenCV  | 281, 500, 3  | array([  0, 128, 192, 224], dtype=uint8) |
| pillow | 281, 500 | array([  0,   1,   2,   3,   4, 255], dtype=uint8) |
| matplotlib  | 281, 500, 3  | array([0., 0.5019608, 0.7529412, 0.8784314], dtype=float32) |

- OpenCV는 이미지 로드할 때 색반전(BRG순서)
- matplotlib는 정규화(0~255)

- PIL로 불러온 이미지를 RGB채널로 보려면 변환해야 함
```
cimg = img.convert('RGB')
imgnp = np.array(cimg)

-----------------------------------

ndarray.shape
before: (281, 500)
after: (281, 500, 3)

np.unique()
before: [  0   1   2   3   4 255]
after: [  0 128 192 224]
```


- `SegmentationObject`와 `SegmentationClass`의 구조
```
(Pdb) np.unique(_mask)
array([  0,   1,   2,   3,   4, 255], dtype=uint8)
(Pdb) np.unique(_cats)
array([  0,   1,  15, 255], dtype=uint8)
```

- 만약 PNG가 color index mapping이라면, `_mask`의 경우는 그냥 순차대로 인덱스 부여한다고 하면 말이 되지만, `_cats`는 분명 클래스 번호이다.
---

### PASCAL VOC Development Kit
- [The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Development Kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION000100000000000000000)

---
서론
- object와 class를 열어본 결과, PNG포맷에 특수한 형태로 데이터가 저장되어 있었다. 즉, 이미지 자체 정보 뿐만이 아니라, object의 수를 count할 수 있게끔 하고, class의 종류를 알 수 있게끔 제작되었다.

---

- `ImageSets/Main`의 `aeroplane_train.txt`를 보면,
    ```
    ...
    2008_000023 -1
    2008_000028 -1
    2008_000033  1
    2008_000036 -1
    2008_000037  1
    2008_000041 -1
    ...
    ```
    이렇게 되어있다. 그런데 하단의 설명을 보면,
    ```
    -1:
    Negative: The image contains no objects of the class of interest. A classifier should give a `negative' output.
    1:
    Positive: The image contains at least one object of the class of interest. A classifier should give a `positive' output.
    0:
    ``Difficult'': The image contains only objects of the class of interest marked as `difficult'.
    ```
    라고 쓰여있다. 그러면 모든 이미지에 대해서 해당 이미지에 해당 클래스가 있는지 -1 또는 1로 모두 리스트에 포함되어야 할 것 같은데, 일부 파일만 들어있다.
    
    물론 train 할 파일만 솎아서 뽑아놓은 것은 이해되지만, 그렇다면 다른 이미지들은 안 쓰이는 것인가?

- example_segmenter.m -> create_segmentations_from_detections.m
    ```
    다음 사용 중 오류가 발생함: create_segmentations_from_detections
    (line 32)
    Could not find detection results file to use to create
    segmentations
    (D:/Users/user2/Downloads/VOCdevkit/results/VOC2012/Main/comp3_det_val_aeroplane.txt
    not found)
    ```

- example_detector.m
    ```
    >> example_detector
    다음 사용 중 오류가 발생함: textread (line 162)
    파일을 찾을 수 없습니다.

    오류 발생: example_detector>train (line 34)
        gtids=textread(sprintf(VOCopts.imgsetpath,VOCopts.trainset),'%s');

    오류 발생: example_detector (line 12)
        detector=train(VOCopts,cls);                            % train detector
    ```

- 내가 필요한건 segmentation

- `ImageSets/Segmentation/train.txt` 파일이 필요하다.
    - 임의로 train할 데이터셋을 선별해야 한다.
        - 총 2,913개 segmentation 데이터 중 train은 1,464개, val은 1,449개로 비율은 약 1:1

    - 그런데 생각해보니 test는 안할꺼니까 (dataset 제작이니까) 전부 포함시키면 될듯

    - lwir과 visbile 두 개가 있지만 파일명은 동일하므로 하나만 만들면 됨.

> create_segmentations_from_detections(id='comp3',confidence=1);

id는 comp3, comp5 등 다양한 게 있다.

- kaist-rgbt는 imageSets 폴더 안에 set00~set11, 그리고 안에 V00n 폴더가 있으므로 이 부분이 살아있도록 이름을 적어둬야 한다.

```
다음 사용 중 오류가 발생함: VOCxml2struct>parse (line 16)
<annotation> closed with <folder>
```

그런데 VOC2012랑 KAIST랑 annotation 파일구조는 같다.

![image](https://user-images.githubusercontent.com/30471027/63775927-56daf180-c91b-11e9-92cf-30f15a4a228e.png)

- 어차피 지금 필요한건 segmentation할 파일에 대한 class 정보니까, xml에서 <annotation><object><name> 여기를 읽어서 매칭해서 저장하면 되지 않을까.

- 사실 webcam_demo.py에서 결과 저장할때 그 PNG array의 값에 매핑한 class number를 적용하면 되겠지만, PNG의 구조가 너무 어렵다.
    - 그러다가 오늘 PASCAL VOC DevKit을 뒤늦게 발견했는데 이게 더 어렵다. 아니 xml에서 막혔다.
### irrelevant
-
---
## TODO
- MATLAB과 친구하기
-
-
---
.

.

.

### TMI
기상시간 7:30

출근시각 12:30

날씨: 오전 비, 구름 조금

외출시간
-

통학시간에 한 것
-
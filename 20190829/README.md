# 20190829

# `pascal voc png`

## DONE
-
-
-
### PACAL VOC
- [Ground truth pixel labels in PASCAL VOC for semantic segmentation
](https://stackoverflow.com/questions/49629933/ground-truth-pixel-labels-in-pascal-voc-for-semantic-segmentation)
    
    Q. 
    ```
    이 PNG파일의 픽셀값은 32의 배수(128, 192, 224) 같아 보이는데, 0과 20 사이 범위를 벗어나지 않습니다.
    픽셀값이랑 GT 라벨이랑 어떤 연관성이 있는지 궁금합니다.
    ```
    comment.
    ```
    저도 최근에 FCN 결과 reproduce했는데 잘 됐습니다. 이미지 어떻게 읽었나요? 리사이즈 했나요? 제가 한번 그랬다가 interpolation이랑 average때문에 라벨 다 망쳤습니다.
    
    ----------

    이거 답 찾으셨나요? 저도 224개 값을 raw byte data로 봤습니다. 근데 color map에서는 보이지가 않아요. 이게 undeined를 의미하는건 아닐까요? 
    VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]] 
    ```

    A1.
    ```
    저도 예전에 이 질문 봤었습니다. 그런데 저도 비슷한 질문을 PASCAL VOC 2012 를 tensorflow deeplab에 시도할 때 했었습니다.

    file_download_and_convert_voc2012.sh 파일을 보면, "#Remove the colormap in the ground truth annotations." 라고 써있습니다. 이 부분은 원본 Segmentation 파일을 처리하고 raw segmented 이미지 파일을 produce하는데, 각 픽셀 값은 0~20 사이를 가집니다.
    ```

    A2.
    ```
    데이터셋 안의 픽셀값은

    0: background
    [1...20] interval : segmented objects, classes [Aeroplane, ..., TVmonitor]
    255: void category, used for border regions (5px) and to mask difficult objects

    윗분의 답은 color palettes와 함께 저장되는 PNG 파일을 discuss합니다. 이건 원래 질문이랑 상관 없는 답 같네요. 링크 결려있는 tensorflow code는 단순히 color map(pallete)이랑 같이 저장된 PNG를 로드하고, ndarray로 변환하고, 다시 PNG로 저장하네요. 수치값은 이 프로세스에서는 안변하고, color palette만 지워집니다.
    ```
- [Python: Use PIL to load png file gives strange results](https://stackoverflow.com/questions/51676447/python-use-pil-to-load-png-file-gives-strange-results?noredirect=1#comment90316222_51676447)

    Q. grayscale로 죽여서 띄웠는데 외곽선만 따서 나옵니다. 왜 그런가요?

    A.
    ```
    문제는 바로 질문자님의 이미지가 palettised되어있기 때문입니다. 그래서, 각 픽셀에 full RGB triplet을 저장하는 것 보다는, 256 RGB triplet list가 있고 각 픽셀 리스트 안에 인덱스를 그냥 저장합니다.

    질문에 답을 드리자면, 네, 맞습니다. palette entry 255는 E0E0C0을 대표합니다. ImageMagick을 이용해서 이미지 디테일을 보고 palette를 dump했습니다.    
    ```

    - im_type `P`가 PNG가 아니라 Palette였음

### [ImageMagick](https://imagemagick.org/script/identify.php)

> magick identify -verbose class.png

```
Image: class.png
  Format: PNG (Portable Network Graphics)
  Mime type: image/png
  Class: PseudoClass
  Geometry: 500x281+0+0
  Units: Undefined
  Colorspace: sRGB
  Type: Palette
  Base type: Undefined
  Endianess: Undefined
  Depth: 8-bit
  ...
  Histogram:
    129545: (  0,  0,  0) #000000 black
      4734: (128,  0,  0) #800000 maroon
       866: (192,128,128) #C08080 srgb(192,128,128)
      5355: (224,224,192) #E0E0C0 srgb(224,224,192)
  Colormap entries: 256
  Colormap:
         0: (  0,  0,  0,255) #000000FF black
         1: (128,  0,  0,255) #800000FF maroon
         ...
         15: (192,128,128,255) #C08080FF srgba(192,128,128,1)         
         ...
         255: (224,224,192,255) #E0E0C0FF srgba(224,224,192,1)        
         ...
```

- `np.unique()`를 했을 때 128, 192, 224만 나오는 이유는 
    1. palette가 32의 배수로 이루어져 있고
    2. 사용된 네 가지 색상의 값에 저 숫자들만 있기 때문
- 색상은 np.random으로 때리는 게 아니라 palette index값을 매칭
- 왜 테두리선이 흰색이 아니냐면 팔레트 공간에서는 최대값(255,255,255)은 32의 배수가 아니므로 32배수의 최대값인 224를 흰색처럼 사용
- 0~7번은 색상 이름이 지어져 있는데, 이는 대표적인 색 조합이기도 하고 자주 사용하게 될(index상 앞부분) 것이므로

- 자세한 결과는 [`magick_identify.txt`](./magick_identify.txt) 참조

- 인터넷에서 아무 PNG 파일이나 받아서 열어보면 dataset과의 차이점은 Class가 dataset은 Pseudoclass이고 온라인은 Directclass 라는 것이다. 그래서 보면 인덱스가 전부 그냥 255로 되어있다.

- 단순히 그림판으로 확장자를 변경하는 것으로는 index가 생성되지 않음.
---

### COCO

- MMDetection model zoo에 있는 segmentation 모델이 COCO 기반이라 class는 80개, Dextr는 Pascal이라 20개

- COCO로 학습한 모델로 PASCAL포맷으로 결과(demo)를 뽑는데 클래스는 80개로 뽑히면...?

- 지금으로선 PASCAL 클래스가 COCO랑 완전히 겹치기를 바라는 수밖에..

- 굳이 PASCAL / COCO class 수를 맞춰야 하는가? 우리는 annotation을 할 것이므로 다다익선.

### irrelevant
-
-
-
---
## TODO
-
-
-
---
.

.

.

### TMI
기상시간 7:30

출근시각 11:00

날씨: 폭우

외출시간
- 점심 (2:30~3:30)

통학시간에 한 것
-

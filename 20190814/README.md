# 20190814

## DONE
- `for ii, line in enumerate(lines):` 부분 다시보기

### DEXTR
- KAIST는 Segmentation Set이 없음. 학습을 위해선 필요할 듯 한데, ... 그냥 annotation 집어넣으면 될까?
- 무슨 생각이었는지, Segmentation 없이 학습되는건 아닌지 라는 말도 안되는 생각을 하며, 하루종일 코드를 한줄 한줄 실행하며 공부함.

![image](https://user-images.githubusercontent.com/30471027/63026566-605d6600-bee6-11e9-8191-5f21cce937a0.png)
- [PASCAL VOC Dataset 구조](https://deepbaksuvision.github.io/Modu_ObjectDetection/posts/02_01_PASCAL_VOC.html)
- paper
    - [Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation](https://arxiv.org/pdf/1808.04818.pdf)
        - 여기서 Segmenation이랑 bbox를 동시에 쓴다는 아이디어를 얻었음.
        - [Illuminating Pedestrians via Simultaneous Detection & Segmentation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Brazil_Illuminating_Pedestrians_via_ICCV_2017_paper.pdf)
            - 여기서 참고했다고 함
    - [Image Segmentation with A Bounding Box Prior](http://vision.stanford.edu/teaching/cs231b_spring1213/papers/lkrs_iccv09_TR.pdf)
        - Segmentation bounding box 검색해서 맨 위에 나온 논문

- [Papers With Code](https://paperswithcode.com/)
    - instance segmentation 뜻을 몰라 검색했는데 맨 위에 나와서 알게됨

- [MMDetection](https://github.com/open-mmlab/mmdetection)
    - 위 사이트에서 instance segmentation 성능 1위찍은 모델
---
### irrelevant
- 학석연계 관련 학점 및 수강 및 과목 등등
---
## TODO
- 서버에 MMDetection 환경 구축
- Segmentation dataset
    - semantic(Segmenationclass)
    - instance(Segmentationobject)
- 이렇게 뽑아낸 Segmentation이 PASCAL set과 호환되게 할 수 있는지 확인할 것. 아마 이미지에 덧씌워 나오는거니 중간에 코드에서 추출해야 할듯.
---
.

.

.

### TMI
기상시간 6:50

출근시각 9:49

날씨: 맑음

외출시간
- 점심식사(14:00~15:00)

통학시간에 한 것
- PR12-004 (~15분)

---

### PASCAL VOC Dataset
```
    - VOCdevkit/VOC2012
        - Annotations
        - ImageSets
            - Action
            - Layout
            - Main
            - Segmentation
                - train.txt
                - train_instances.txt
                - trainval.txt
                - val.txt
                - val_instances.txt
        - JPEGImages
        - SegmentationClass
        - SegmentationObject
```

### KAIST-RGBT

```
- annotation_json
- annotation_jsons
- annotation_sml
- annotations
- annotations-xml-15
- annotations-xml-181027
- annotations_vbb
- cyc
- cyc2
- images
    - set00
        - V000
            - lwir
            - visible
        - V001
        - V002
        - V003
        - V004
        - V005
        - V006
        - V007
        - V008
    - set01
    - set02
    - set03
    - set04
    - set05
    - set06
    - set07
    - set08
    - set09
    - set10
    - set11

- imageSets
    - jwkim-edit-code.txt
    - jwkim-video-01.txt
    - test-all-01-set06-V000.txt
    - test-all-01.txt
    - test-all-20.txt
    - test-day-01.txt
    - test-day-20.txt
    - test-night-01.txt
    - test-night-20.txt
    - train-all-02.txt
    - train-all-04.txt
    - train-all-20.txt
    - train-day-02.txt
    - train-day-04.txt
    - train-day-20.txt
    - train-night-02.txt
    - train-night-04.txt
    - train-night-20.txt
    - video-all-01-set08-V0001.txt
- test

- coco.py
- eval_MR_multisetup.py
- xml_to_json_dchan.py

- gfriend.json
- jwkim_debug_annotations.json
- kaist_annotations_test20.json
- kaist_annotations_test20_2015.json
- Sejong_RCV_total_test.json
- test_all-01.json
- test_all-01_0709.json
- test_all.json
- test_all_-01.json
- test_all_20.json
- test_alls-01.json
- test_alls.json
- test_alls_20.json
- train_all.json
- train_alls.json
```
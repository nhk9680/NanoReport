# Troubleshooting summary
sorted by date

## DONE
- DGX SSD6T 용량 오류
    - 현재 원인을 알 수 없음. 오전에 특별히 작업한 것이 없는데 오후에 가용량이 10GB(11.x->22.x) 증가함
- docker 메뉴얼: 삭제 편 작성
- 서버에서 demo.py의 plt가 작동하지 않아 로컬에서 시도
    - 리눅스 sh파일 때문에 VMware에서 환경 구축중...잘나옴
    - 당연히 서버 터미널에서 실행하면 GUI를 띄울 수 없으므로 안된다. 근데 분명 지난번 sumemr school때는 Moba로 띄웠는데 ...
    - 아무튼 우리 데이터셋 적용은 이게 보이든 안보이든 서버에서 작업할 수 있다.
- windows에서 .sh 파일 실행하기
    - 프로그램 및 기능 -> Windows 기능 켜기/끄기 에서 `Linux용 Windows 하위 시스템` 체크 후 리부팅


---

- docker run 커맨드 숙지(삽질)
    - docker run 시 `--user` 에는 이름이 아니라 id값(100x)를 써야 인식됨.    
    - Unable to find user root: no matching entries in passwd file
        - Dockerfile에 USER 선언 하지말고 그냥 만들기
    - i have no name!
        - `--user=0`(root) 로 들어가서 passwd에 user 추가
    - linux spec user unable to find user
        - 골치아프니 root로 도커 접속
        
- 리눅스 서버 사용중인 포트 조회 `netstat -nap | grep 0.0.0.0:88 | sort `

- MMDetection이 sejong-rcv에 fork된 것을 방금(2019/08/16 03:41) 확인

---

- 2기 서버교육
- 김병주 연구원 IP 미등록 건
    - 이외 추가 부여받은 5개 IP들을 DGX 서버에 추가 요청
- MobaXTerm docker 별 매크로 설정
- 서버 `history` 로그에 Time 추가

---

- kaist rgbt annotations에는 15랑 18버전이 있는데, 후자가 더 정확도가 개선된 파일이다.

---

<br>
<br>
<br>

---

## Python

- `*args`, `**kargs`
    - `*args` : parameter들을 tuple 형태로 입력
    - `**kargs` : parameter들을 dict(key, value) 형태로 입력
- `isinstance(value, type)` : value의 type이 type인가 ? Y : N;
- `@decorator` : 함수(클래스)의 실행,선언,정의 전후로 실행해주는 말 그대로 데코레이터
- python은 class나 function이나 module을 parameter로 줄 수 있다.

---

- len(list)=n일 때, list[:n+a] == list[:n]
- `with torch.no_grad():` no gradient, 학습, 추적, 메모리 저장 안함

---

```
- `if A or B` 에서, A의 조건이 만족되지 못한 상황에서, B 함수 내부에 breakpoint를 걸었지만(pdb.set_trace(), step접근 모두) 걸리지 않았음.
- 그러나 B조건함수 내부에서 정의되는 `self.obj_dict` 변수는 if문 통과 후 성공적으로 정의됨
-  두 조건의 순서를 변경하여 해결 시도
    - 그래도 A를 먼저 접근함.
    - 코드에서 A함수가 B보다 위에 써있음. 그래서 바꿈.
    - 그래도 안됨. 마치 컴파일러는 답을 알고 있다는 듯이.
    - 어차피 조건 만족 못하니 A조건 삭제하고 접근
```
- 코드 착각. A함수 내에 변수정의가 있었음
- 그래도 위 상황은 이해가 안됨

---


<br>
<br>
<br>

---

## DEXTR

- 직접 모델 학습을 위해선 `KerasTensorFlow`가 아닌 `PyTorch` 버전을 이용해야 함.
- `demo.py` 실행시 `extreme_points_ori = np.array(plt.ginput(4, timeout=0)).astype(np.int)`에서 멈춤. 정확히는 GUI가 실행되지만 보이지 않는 상태.

---

- KAIST Dataset이 COCO 형태인 것 같아, coco pretrained를 먼저 적용해보고자 함. coco.py에 data 파라미터 자체가 없는지 안넘어가서 NameError가 뜸. 반년째 데이터셋 적용을 못해보는 상황
- torch.data.utils에서 불러오는 건데 pascal.py에서는 잘 데려오면서 coco는 못한다. 왜?
- 아무리 구글링을 해도 데이터셋 적용하는 글이 없다. 물론 내가 못찾는 것이라 예상.
- Tensorflow tfrecord 삽질은 1달이 걸렸는데 이건..
- 시작이 반이지만 시작을 못하고 있다. 환경설정만 완료. 데모만 돌림

---

- COCO 데이터셋은 KAIST 셋을 변환해야 하므로 우선 PASCAL VOC로 진행하기로 하고, 기존의 PASCAL 코드 실행.

`train_pascal.py` 실행 중 line 19에서
```
AttributeError: 'scipy.misc' object has no attribute 'imsave'
```

오류가 발생해서 pillow, scipy 삭제 후 재설치. [솔루션](https://github.com/oduerr/dl_tutorial/issues/3)

재설치 후 torchvision과 torch 버전이 꼬여서 ImportError 발생. [솔루션](https://github.com/facebookresearch/maskrcnn-benchmark/issues/226) 찾아서 업데이트 후 해결.

그러나 재설치 했음에도 여전히 같은 오류 발생. 추가 구글링

---

- scipy.misc imsave 오류 해결
    - 1.0 미만 버전(0.19.1)로 다운그레이드 하여 메소드 복원.
    - 그러나 본 코드에서는 단지 imsave 하나만 사용하기 위해 설치하는데, 너무 무거운 걸 끌어다 쓰는 건 아닌지...
    - 차라리 그냥 from cv2 import imsave 하는게 나았을수도
- train_pascal.py 실행 성공
    - 이유는 모르겠으나 다 실행된 후
        ```
        The program finished and will be restarted
        ```
        하면서 재시작됨. 뭐하러?
    - result.txt 참고

---

- GPU->CPU 세팅을 했음에도 local에서 실행 시 RuntimeError가 발생해서 서버에서 작업중.
- > voc_train = pascal.VOCSegmentation(split='train', transform=composed_transforms_tr)

- ```python
    self.obj_list_file = os.path.join(
        self.root, self.BASE_DIR, 'ImageSets', 'Segmentation', '_'.join(self.split) + '_instances' + area_th_str + '.txt')

    _splits_dir = os.path.join(_voc_root, 'ImageSets', 'Segmentation')
    ```
    -> `~\VOC2012\ImageSets\Segmentation\train.txt`
        
            2007_000033
            2007_000042
            2007_000061
            2007_000123
            2007_000129
            ...

- `for splt in self.split:` 왜 for문을 사용했을까? self.split은 `'train'`이다.

---

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

- 현재 계획
    1. MMDetection Demo 실행
    2. Image output 부분 코드에서 segmentation 파트 찾기
    3. 해당 파트가 json포맷 로드 형태일 경우 output json파일 확인
        3-1. 파일 포맷이 PASCAL VOC or COCO 일 경우: 바로 적용가능
        3-2. 아니면 해당 포맷으로 변환
    4. 해당 파트가 기타 다른 형태일 경우
        - 이 부분만 붙잡고 일주일 넘게 걸릴 듯 하니, 다른 연구원들에게 도움 요청
    5. instance segmentation dataset 확보 완료
    
    - 우려사항
        - domain adaptation이 이루어지지 못하므로 결과가 좋지 않을 것.
        - 또한 완벽한 GT가 아니므로 학습 시에도 퀄리티가 떨어질 듯함.
    
    - 예상 
    
- 다음 계획
    - semantic segmentation dataset 같은 원리로 제작
    - 어제 Dextr 논문을 다시 읽었을 때, 원리가 적층식, 즉 Class 위에 Object가 쌓이는 형태로 이해했음.
- MMDetection 세팅
    - 10분이면 끝날 줄 알았는데 늘어짐
    - 처음에 conda로 환경을 구축하려 했으나, 많은 오류로 인해 포기하고 제공되는 도커파일 선택

---

- `train_instances.txt` 에는 `{"이미지이름":[클래스1,클래스2, ...]}` 형태로 데이터가 들어있음.

- 이게 없으면 전처리 과정을 거쳐 파일을 생성

```python
    def _preprocess(self):
        self.obj_dict = {}
        obj_counter = 0
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            _mask = np.array(Image.open(self.masks[ii]))
            _mask_ids = np.unique(_mask)
            if _mask_ids[-1] == 255:
                n_obj = _mask_ids[-2]
            else:
                n_obj = _mask_ids[-1]

            # Get the categories from these objects
            _cats = np.array(Image.open(self.categories[ii]))
            _cat_ids = []
            for jj in range(n_obj):
                tmp = np.where(_mask == jj + 1)
                obj_area = len(tmp[0])
                if obj_area > self.area_thres:
                    _cat_ids.append(int(_cats[tmp[0][0], tmp[1][0]]))
                else:
                    _cat_ids.append(-1)
                obj_counter += 1

            self.obj_dict[self.im_ids[ii]] = _cat_ids

        with open(self.obj_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.obj_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.im_ids[ii], json.dumps(self.obj_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing finished')
```        

> _mask_ids = np.unique(_mask)
>
> (Pdb) _mask_ids
>
> array([  0,   1,   2,   3,   4, 255], dtype=uint8)

- `np.unique` : 중복되는 배열 원소를 제거.

- `n_obj` : 오브젝트의 개수. 

-  `np.where(값)` : 값의 위치(index)를 반환. 2차원이면 (array, array) 이렇게

> `_cat_ids.append(int(_cats[tmp[0][0], tmp[1][0]]))`

- 마스크에서의 픽셀 좌표를 그대로 카테고리에 옮겨서 찍으면 카테고리 이미지에서 그 픽셀에는 그 카테고리의 클래스정보가 들어있겠지요?

- ...그런데 주말에 추출한 데이터셋에는 class 정보가 없다.

```
(Pdb) np.unique(_mask)
array([  0,   1,   2,   3,   4, 255], dtype=uint8)
(Pdb) np.unique(_cats)
array([  0,   1,  15, 255], dtype=uint8)
```

- 마스크 부분 뽑아서 카테고리에 점찍고 알아낸다. 해서 각 Object가 어떤 Class인지 페어링한다.

    그런데 이미지값은 분명 RGB가 있는데, 어떻게 1채널?

    -   몰라도 그냥 `color_mask` 부분 수정해서 만들 순 있다.

    설마 PNG라서?

---

<br>
<br>
<br>

---

## MMDetection

- mmdetection anaconda 세팅
- COCO, KAIST rgbt, pkl 구조 다시 공부
    - instance_val2017.json 에는 segmentation data가 들어있음.
- model zoo에서 가장 성능이 잘나온 `Hybrid Task Cascade (HTC)`의 `X-101-64x4d-FPN`의 DCN 버전 pretrained model 다운로드
    - `stuffthingmaps` 라는 추가 데이터가 필요해서, 복잡해서 패스
    - 대신 `X-101-64x4d-FPN` 모델 다운로드해서 `test.py` 실행해서 `pkl`, `json` 파일 획득
        - 용량은 수십MB, bbox와 mask json파일 용량의 합이 대략 results.pkl과 같음
    - 그러나 시각적으로 결과를 보는 방법을 모름.
    - 못보더라도 어떤 데이터 형태인지 분석이 필요. 그래야 이 output을 적용할 수 있는지, 또는 어떻게 가공해야 하는지 결정할 수 있음.
    - 그런데 필요한 건 mAP 같은 게 아니라 단순히 segmentation 데이터인데, annotation+train 파일을 불러와서 mAP를 뽑으려 함.
    - 단순히 결과만 뽑는 demo가 필요함

- VNC에서 `show` 옵션으로 `test.py` 실행했으나 GUI 뜨지 않음.
    - `webcam_demo.py` 코드 확인해보니 `cv2.videocapture()`라서 동영상 먹여보기로 함.

- Mask R-CNN으로 전향 고민
- `mmcv` 모듈로 pkl을 dump하길래 이걸로 load하면 복원할 수 있지 않을까 해서 공부

- > The annotation of a dataset is a list of dict, each dict corresponds to an image.
There are 3 field `filename` (relative path), `width`, `height` for testing,
and an additional field `ann` for training. `ann` is also a dict containing at least 2 fields:
`bboxes` and `labels`, both of which are numpy arrays. Some datasets may provide
annotations like crowd/difficult/ignored bboxes, we use `bboxes_ignore` and `labels_ignore`
to cover them.

    이걸 보면, bbox와 label, 즉 annotation data가 없어도 되는 듯? 근데 그러면 말이 안 된다.

- `webcam_demo.py` 수정
    - videocapture() 에 임의의 비디오 삽입
    - `inference_detector`의 output을 pkl로 저장
    - `show_results` 의 `out_file=None` 켜기

- [[MMDetection] 논문 정리 및 모델 구현](https://wordbe.tistory.com/43)

---

- `webcam_demo.py` 결과 비디오로 제작
    - Mask R-CNN 기반이여서 그런지 멀리 있는건 잘 못잡음.
- result의 array와 counts 값이 무엇이고 어떻게 들어오는지 알기 위해 step-by-step 디버깅
    
    - `self.forward()` 는
        >
            [docs]    def forward(self, *input):
            r"""Defines the computation performed at every call.

            Should be overridden by all subclasses.

            .. note::
                Although the recipe for forward pass needs to be defined within
                this function, one should call the :class:`Module` instance afterwards
                instead of this since the former takes care of running the
                registered hooks while the latter silently ignores them.
            """
            raise NotImplementedError
        
        >
            모든 call에서 수행된 계산을 정의합니다. 모든 subclass에 의해 override되어야 합니다. 이 함수 내에서 forward pass 방법이 정의되어야 하지만, 후자가 조용히 무시하는 동안 전자가 register된 hooks를 실행하는 것을 처리하기 때문에, 이 기능 대신 `torch.nn.Module` 인스턴스를 나중에 호출해야야 합니다.

        너무 어렵다.
        
        반환값이 정의되지 않은 함수에서 핵심 데이터를 반환받는다. 무언가 소스코드가 생략되어 있는 듯 하다.

        > PyTorch에서 torch.autograd.Function 의 서브클래스(subclass)를 정의하고 forward 와 backward 함수를 구현함으로써 쉽게 사용자 정의 autograd 연산자를 정의할 수 있습니다. 그 후, 인스턴스(instance)를 생성하고 함수처럼 호출하여 입력 데이터를 포함하는 Variable을 전달하는 식으로 새로운 autograd 연산자를 쉽게 사용할 수 있습니다.
        
        [(출처: 예제로 배우는 PyTorch)](https://9bow.github.io/PyTorch-tutorials-kr-0.3.1/beginner/pytorch_with_examples.html#pytorch-autograd)

        아무튼 그런데 왜 fp16의 decorators로 넘어가는지 모르곘다. 아마 원본 소스코드에 @가 붙어있어서 그런 듯 하다.
        - fp16 = Floating Point 16 bit

        찾았다.

        demo, 즉 test 과정이니 loss를 학습하지 않으므로 `return_loss=False`

    ```python
    (Pdb) w
    /mmdetection/demo/webcam_demo.py(56)<module>()
    -> main()
    /mmdetection/demo/webcam_demo.py(40)main()
    -> result = inference_detector(model, img)
    /mmdetection/mmdet/apis/inference.py(67)inference_detector()
    -> return _inference_single(model, imgs, img_transform, device)
    /mmdetection/mmdet/apis/inference.py(94)_inference_single()
    -> result = model(return_loss=False, rescale=True, **data)
    /opt/conda/envs/open-mmlab/lib/python3.7/site-packages/torch/nn/modules/module.py(547)__call__()
    -> result = self.forward(*input, **kwargs)
    /mmdetection/mmdet/core/fp16/decorators.py(49)new_func()
    -> return old_func(*args, **kwargs)
    /mmdetection/mmdet/models/detectors/base.py(88)forward()
    -> return self.forward_test(img, img_meta, **kwargs)
    /mmdetection/mmdet/models/detectors/base.py(79)forward_test()
    -> return self.simple_test(imgs[0], img_metas[0], **kwargs)
    /mmdetection/mmdet/models/detectors/cascade_rcnn.py(274)simple_test()
    -> x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
    /mmdetection/mmdet/models/detectors/test_mixins.py(10)simple_test_rpn()
    -> proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
    /mmdetection/mmdet/core/fp16/decorators.py(127)new_func()
    -> return old_func(*args, **kwargs)
    /mmdetection/mmdet/models/anchor_heads/anchor_head.py(221)get_bboxes()
    -> scale_factor, cfg, rescale)
    > /mmdetection/mmdet/models/anchor_heads/rpn_head.py(67)get_bboxes_single()
    -> assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]

    ```

    <ROI 5개인 이유를 찾아라>
    [if self.num_outs > len(outs):](https://github.com/open-mmlab/mmdetection/blob/0b96173149603abed1feeea4fe814b9fab383787/mmdet/models/necks/fpn.py#L121)

    self.num_outs = 5
    len(outs) = 4

    여기서 레이어 하나 더 추가해서
    tuple(outs) 반환

    어느 순간 tensor([1000, 5]) 형태가 뜨니, 1000을 찾자.
    > {'nms_across_levels': False, 'nms_pre': 1000, 'nms_post': 1000, 'max_num': 1000, 'nms_thr': 0.7, 'min_bbox_size': 0}
    
    [get_bboxes_single](https://github.com/open-mmlab/mmdetection/blob/0b96173149603abed1feeea4fe814b9fab383787/mmdet/models/anchor_heads/rpn_head.py#L55)

    (https://github.com/open-mmlab/mmdetection/blob/0b96173149603abed1feeea4fe814b9fab383787/mmdet/models/anchor_heads/rpn_head.py#L78)

    이제 1000, 5가 되는 부분을 찾자.

    > _, topk_inds = scores.topk(cfg.nms_pre)

    `Torch.topk` 명령어 써로 큰것만 뽑아서 [163200,4]에서 [1000,4] 로 줄어든다.

    proposal이 [1000,5] 모양이였으니까 이걸 만드는 dealta2bbox 함수를 찾아가면 transforms.py
    
    [delta2bbox](https://github.com/open-mmlab/mmdetection/blob/0b96173149603abed1feeea4fe814b9fab383787/mmdet/core/bbox/transforms.py#L34)

    그냥 여기서 잘라버림.

    [num = min(cfg.max_num, proposals.shape[0])](https://github.com/open-mmlab/mmdetection/blob/0b96173149603abed1feeea4fe814b9fab383787/mmdet/models/anchor_heads/rpn_head.py#L101)

    ---

- [`proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)`]
- tensor([1000,4]) 를 tensor([1000,5]) 로 늘려준다.
- proposals[0]
    - before
    - `tensor([1059.0000,  0.0000,      1059.0000,  13.1144                 ], device='cuda:0')`
    - after
    - `tensor([1.0590e+03, 0.0000e+00,  1.0590e+03, 1.3114e+01, 5.6994e-01  ], device='cuda:0')`
- 앞의 네 개는 [x1, y1, x2, y2] 이고, 뒤에 하나는 `scores`
- [`rois = bbox2roi(proposal_list)`](https://github.com/open-mmlab/mmdetection/blob/0b96173149603abed1feeea4fe814b9fab383787/mmdet/models/detectors/cascade_rcnn.py#L286)
    ```python
    (Pdb) proposal_list[0][0]
    tensor([540.9440, 391.5113, 665.9113, 769.5121,   0.9999], device='cuda:0')
    (Pdb) rois[0]
    tensor([  0.0000, 540.9440, 391.5113, 665.9113, 769.5121], device='cuda:0')
    ```
    ```python
    def bbox2roi(bbox_list):
        """Convert a list of bboxes to roi format.
        Args:
            bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
                of images.
        Returns:
            Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
        """
    ```

- resutls=[ms_bbox_result, ms_segm_result]

- `bbox_result = bbox2result(det_bboxes, det_labels, bbox_head.num_classes)`

    ```python
    def bbox2result(bboxes, labels, num_classes):
        """Convert detection results to a list of numpy arrays.
        Args:
            bboxes (Tensor): shape (n, 5)
            labels (Tensor): shape (n, )
            num_classes (int): class number, including background class
        Returns:
            list(ndarray): bbox results of each class
        """
        if bboxes.shape[0] == 0:
            return [
                np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
            ]
        else:
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
            return [bboxes[labels == i, :] for i in range(num_classes - 1)]
    ```

    labels가 살아있는 bboxes만 살려서 넘김, 나머진 빈 값으로 넘김

    - [rle = mask_util.encode(np.array(im_mask[:, :, np.newaxis], order='F'))[0]](https://github.com/open-mmlab/mmdetection/blob/0b96173149603abed1feeea4fe814b9fab383787/mmdet/models/mask_heads/fcn_mask_head.py#L177)

    ```python
    # encode mask to RLEs objects
    # list of RLE string can be generated by RLEs member function
    def encode(np.ndarray[np.uint8_t, ndim=3, mode='fortran'] mask):
        h, w, n = mask.shape[0], mask.shape[1], mask.shape[2]
        cdef RLEs Rs = RLEs(n)
        rleEncode(Rs._R,<byte*>mask.data,h,w,n)
        objs = _toString(Rs)
        return objs
    ```

    구한 segmentation을 pycocotools를 이용해 인코딩한다. 인코딩하기 전 모습은 (w,h,1) binary mask이다.

---

- `decode()` 로 복원 성공
- GT는 무조건 xml,csv,txt 같은 string 형태의 데이터 포맷이어야 한다는 편견을 갖고 있었음.

- pascal은 그냥 색칠된 이미지로 제공되고 있었음.

- img.shape=(228,302,3)
- mask.shape=(228,302)
- img[mask].shape=(3131,3)
    - numpy.ndarray[numpy.ndarray]
    - img[(228,302)]
    - 직관적으로는 이해가 되지만(1:1 비율 반투명), 코드상으로는 이해가 되지 않는다.
    - mask가 bool type이여서 마스크 부분만 활성화해주는 것같다.

### *흰색의 값은 1이 아니고 255다. (feat. 김지원 연구원님)*

---

- 데이터셋 제작: 95352개의 segmentation image (진행중)
    - 파일 I/O 삽질

---

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
    ```matlab
    다음 사용 중 오류가 발생함: create_segmentations_from_detections
    (line 32)
    Could not find detection results file to use to create
    segmentations
    (D:/Users/user2/Downloads/VOCdevkit/results/VOC2012/Main/comp3_det_val_aeroplane.txt
    not found)
    ```

- example_detector.m
    ```MATLAB
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

```MATLAB
다음 사용 중 오류가 발생함: VOCxml2struct>parse (line 16)
<annotation> closed with <folder>
```

그런데 VOC2012랑 KAIST랑 annotation 파일구조는 같다.

![image](https://user-images.githubusercontent.com/30471027/63775927-56daf180-c91b-11e9-92cf-30f15a4a228e.png)

- 어차피 지금 필요한건 segmentation할 파일에 대한 class 정보니까, xml에서 <annotation><object><name> 여기를 읽어서 매칭해서 저장하면 되지 않을까.

- 사실 webcam_demo.py에서 결과 저장할때 그 PNG array의 값에 매핑한 class number를 적용하면 되겠지만, PNG의 구조가 너무 어렵다.
    - 그러다가 오늘 PASCAL VOC DevKit을 뒤늦게 발견했는데 이게 더 어렵다. 아니 xml에서 막혔다

---

- PNG RGB convert 과정을 보기 위해 convert 메소드를 열어보았으나 `ImagingCore` 클래스를 step할 수 없어 중단
- object와 class의 np.unique() 값이 [0,128,192,224]로 같음. 즉 각 index에 1:1 매핑이 아니라는 것. 또한 color 값이 아니라는 것.

- np.unique() 했을때 [0,1,15,255] 가 나오는데,

> alphabetical order (1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car , 8=cat, 9=chair, 10=cow, 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor)

- 이렇게 보았을 때 1 비행기 맞고 15 사람 맞다.

- [PNG Decoder](https://github.com/lvandeve/lodepng) 를 이용해서, 어느 부분에 class number가 매핑되는지 알아볼 예정.

### PASCAL VOC Development Kit

``` Matlab
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

---

### ThermalGAN
- 데이터셋 다운로드 링크 메일로 전해받음
- ThermalGAN의 SegmentationObject PNG Colorspace는 Grayscale임. 그래서 #010101과 #020202는 육안으로 구분 불가능.

---

- `pascal.py` 수정
    ```python
    BASE_DIR = 'ThermalWorld_VOC_v1_0/dataset/train'

    _image_dir = os.path.join(_voc_root, 'ThermalImages')

    _image = os.path.join(_image_dir, line + ".npy")
    ```


obj_list_file은 .txt로 받았음
_check_preprocess 함수에서 파일 리스트 검사했을 때 파일이 있으면 로드하는데, 할때 json.load로 함.

?????

근데 기본 pascal 데이터셋으로는 잘 돌았음.

???????

롤백 분석결과
json.load는 file이 txt여도 inner format이 json이면 상관없이 읽어들인다.

```python
File "/raid/nhkim/jobs/DEXTR-PyTorch/dataloaders/pascal.py", line 173, in _preprocess
    _cat_ids.append(int(_cats[tmp[0][0], tmp[1][0]]))
TypeError: only size-1 arrays can be converted to Python scalars
```

```python
(Pdb) _cats[tmp[0][0], tmp[1][0]]

array([125,   0,   0], dtype=uint8)
```
```python
------------original----------------
(Pdb) _cats.shape
(281, 500)

(Pdb) _cats[tmp[0][0], tmp[1][0]]
1

(Pdb) _mask.shape
(281, 500)

tmp
(array([106, 106, 106, ..., 182, 182, 182]), array([234, 235, 236, ..., 219, 220, 224]))


-------ThermalWorld-------------

(Pdb) _cats.shape
(889, 1185, 3)

(Pdb) _cats[tmp[0][0], tmp[1][0]]
array([125,   0,   0], dtype=uint8)

(Pdb) _mask.shape
(889, 1185)

(Pdb) tmp
(array([360, 360, 360, ..., 423, 424, 424]), array([480, 481, 482, ..., 522, 521, 522]))
```

차원 축소가 필요해서 보니까, 125번은 class도 아니고 열어보니 colormap도 없으며 color도 통일이 안되고 번져있다. 그래서 노이즈 클래스가 잔뜩 들어있다. 매핑 기준을 모르겠음

=> 불가능, 기본 COCOdataset을 이용해 COCOSegmentation Class로 적용해보고, 잘 도면 mmdet 추출
# 20190820

## DONE
### Python
- `*args`, `**kargs`
    - `*args` : parameter들을 tuple 형태로 입력
    - `**kargs` : parameter들을 dict(key, value) 형태로 입력
- `isinstance(value, type)` : value의 type이 type인가 ? Y : N;
- `@decorator` : 함수(클래스)의 실행,선언,정의 전후로 실행해주는 말 그대로 데코레이터
- python은 class나 function이나 module을 parameter로 줄 수 있다.
### MMDetection
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
        
        >   모든 call에서 수행된 계산을 정의합니다. 모든 subclass에 의해 override되어야 합니다.

        > 이 함수 내에서 forward pass 방법이 정의되어야 하지만, 후자가 조용히 무시하는 동안 전자가 register된 hooks를 실행하는 것을 처리하기 때문에, 이 기능 대신 `torch.nn.Module` 인스턴스를 나중에 호출해야야 합니다.

        너무 어렵다.
        
        반환값이 정의되지 않은 함수에서 핵심 데이터를 반환받는다. 무언가 소스코드가 생략되어 있는 듯 하다.

        > PyTorch에서 torch.autograd.Function 의 서브클래스(subclass)를 정의하고 forward 와 backward 함수를 구현함으로써 쉽게 사용자 정의 autograd 연산자를 정의할 수 있습니다. 그 후, 인스턴스(instance)를 생성하고 함수처럼 호출하여 입력 데이터를 포함하는 Variable을 전달하는 식으로 새로운 autograd 연산자를 쉽게 사용할 수 있습니다.
        
        [(출처: 예제로 배우는 PyTorch)](https://9bow.github.io/PyTorch-tutorials-kr-0.3.1/beginner/pytorch_with_examples.html#pytorch-autograd)

        아무튼 그런데 왜 fp16의 decorators로 넘어가는지 모르곘다. 아마 원본 소스코드에 @가 붙어있어서 그런 듯 하다.
        - fp16 = Floating Point 16 bit

        찾았다.

        demo, 즉 test 과정이니 loss를 학습하지 않으므로 `return_loss=False`

    ```
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

### irrelevant
- 수강신청
- 창의학기제 OT
---
## TODO
- result 분석 마무리
---
.

.

.

### TMI
기상시간 6:30

출근시각 10:00

날씨: 맑음

인터넷 검색 기록
- 

외출시간
- 점심식사 (12:30~13:30)

통학시간에 한 것
- 고전독서
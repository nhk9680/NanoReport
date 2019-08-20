# 20190819

## DONE
- 2기 서버교육
- 김병주 연구원 IP 미등록 건
    - 이외 추가 부여받은 5개 IP들을 DGX 서버에 추가 요청
- MobaXTerm docker 별 매크로 설정
- 서버 `history` 로그에 Time 추가
### ~~DEXTR~~ MMDetection
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
### irrelevant
- 대학원 문의
    - 학석공동과목 상관없이 수강가능
    - 추후 학과장을 통해 수료과목으로 인정 요청
- 중고나라 봇 수리
---
## TODO
- 수강신청
- `show_results` 를 img 저장으로 교체하기
---
.

.

.

### TMI
기상시간 7:30

출근시각 11:00

날씨: 일교차 큼

웹서핑
- kakao online 코딩테스트
- pyinstaller setting icon

통학시간에 한 것
- 고전독서

---

한 게 없을수록 문장이 길어진다.
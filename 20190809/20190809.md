# 20190809

## DONE

- [Dataset_to_VOC_converter](https://github.com/CasiaFan/Dataset_to_VOC_converter)

### DEXTR

- COCO 데이터셋은 KAIST 셋을 변환해야 하므로 우선 PASCAL VOC로 진행하기로 하고, 기존의 PASCAL 코드 실행.

- `train_pascal.py` 실행 중 line 19에서
```
AttributeError: 'scipy.misc' object has no attribute 'imsave'
```
오류가 발생해서 pillow, scipy 삭제 후 재설치. [솔루션](https://github.com/oduerr/dl_tutorial/issues/3)

재설치 후 torchvision과 torch 버전이 꼬여서 ImportError 발생. [솔루션](https://github.com/facebookresearch/maskrcnn-benchmark/issues/226) 찾아서 업데이트 후 해결.

그러나 재설치 했음에도 여전히 같은 오류 발생. 추가 구글링

---
### irrelevant
- 관심과목담기 마지막날
---
## TODO
- Daily
- 고전독서
---
.

.

.

### TMI
기상시간 6:45

출근시각 9:00 -> 12:00 -> 3:00
- 사유: 알바 대타(어린이회관 교육)

구름 조금, 여우비

인터넷 검색 기록
- 

외출시간

통학시간에 한 것
- 잠
# 2019

## DONE
-
### ThermalGAN

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

```
File "/raid/nhkim/jobs/DEXTR-PyTorch/dataloaders/pascal.py", line 173, in _preprocess
    _cat_ids.append(int(_cats[tmp[0][0], tmp[1][0]]))
TypeError: only size-1 arrays can be converted to Python scalars
```

```
(Pdb) _cats[tmp[0][0], tmp[1][0]]

array([125,   0,   0], dtype=uint8)
```
```
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

---
### irrelevant
---
## TODO
- RAID 이관
- 용량 제한
    - Linux fdisk 공부
- WOL 설정
    - 업데이트 등으로 자꾸 리부팅되면 손을 못씀
---
.

.

.

### TMI
기상시간 7:00

출근시각 13:43

날씨: 구름 많음
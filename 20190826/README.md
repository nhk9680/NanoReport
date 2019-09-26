# 20190826

## DONE
-
-
### Python

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
### Dextr
- `train_instances.txt` 에는 `{"이미지이름":[클래스1,클래스2, ...]}` 형태로 데이터가 들어있음.

- 이게 없으면 전처리 과정을 거쳐 파일을 생성

```
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
### irrelevant
-
---
## TODO
- PNG 포맷 공부
---
.

.

.

### TMI
기상시간 6:30

출근시각 11:30

날씨: 구름 조금 

외출시간
- 고전독서(30)

통학시간에 한 것
- 고전독서
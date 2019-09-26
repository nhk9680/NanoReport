# 20190813

## DONE
- Docker migration(진행중)
    - daemon.json 파일 문제로 이전 도커 복원
- Anaconda X PyCharm
    - ImportError 해결 (PATH 문제)
- Background process
    - `Ctrl`+`z` 하면
        > [1]+  Stopped                 sudo rsync -vaxP /var/lib/docker /raid2/docker

        하면서 백그라운드로 넘어가고, 다시 보려면 `fg %1` 하면 돌아옴.
- > tensorboard --logdir=`파일이 위치한 폴더 경로`
---
### Daily

- [ ] PR12
- [ ] PD Paper Reading

### DEXTR
- GPU->CPU 세팅을 했음에도 local에서 실행 시 RuntimeError가 발생해서 서버에서 작업중.
- > voc_train = pascal.VOCSegmentation(split='train', transform=composed_transforms_tr)

- ```
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
### irrelevant
-
---
## TODO
- `for ii, line in enumerate(lines):` 부분 다시보기
---
.

.

.

### TMI
기상시간 6:45

출근시각 10:30

날씨: 구름 조금

인터넷 검색 기록
- 

외출시간
- 점심식사(13:00~14:00)

통학시간에 한 것
- 고전독서
# 20190822

> ## TODO
>   - [x] `decode.py` 실행해서 복원해보기
>
>   - COCO와 VOC의 Segmentation format 비교
>       - 다르면 COCO로 우선 뽑고 COCO2VOC 구글에 많으니 그거로 돌리기
>   - `.pkl`
>   - [x] 시간표 최종결정하기

## DONE
- 
-
-
### MMDetection
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

# 흰색의 값은 1이 아니고 255다. (feat. 김지원 연구원님)

---
### irrelevant
-
---
## TODO
- 데이터셋 뽑아내기
---
.

.

.

### TMI
기상시간 7:30

출근시각 10:20

날씨: 구름 조금

인터넷 검색 기록
- 

외출시간
- 점심식사 (13:30~14:30)

통학시간에 한 것
- 고전독서
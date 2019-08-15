# 20190815

> Paper Reading 슬라이드 정리 혹은 이슈로 정리 해주셈
> KAISTPD_VBB2VOC.py 파일 기대할께

- 이걸 오늘 봤습니다... 

---

## DONE
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
### DEXTR
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
-
---
### irrelevant
-
---
## TODO
- 대학원교학과 or 수업과 문의
    - 타학과 (대학원)학석연계 과목 신청시 인정?
        -ex) 지능기전 -> 컴공 연계과목
    - 타학과 과목 신청시 인정?
        -ex) 지능기전 -> 컴공 공통과목
    - 대학원생이 학부과목 수강시 수료학점 인정?
        -ex) 대학원생 -> 지능기전 딥러닝시스템 / 인턴쉽2 / ...
- WeeklyReport 작성
- PaperReading 슬라이드 정리
- VBB2VOC.py 만들기
---
.

.

.

### TMI
기상시간 10:00

출근시각 11:30(재택근무)

실근무시간 3+3

날씨: 비 후 그침
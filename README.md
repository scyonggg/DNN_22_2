# 2022-2 DNN Mid-term project implementation code
- Subject: Bag of Tricks for Image Classification
# Quick Start

## Requirements
  ```
  timm
  wandb
  pytorch
  kornia
  ```

## Steps
**0. train.py의 parser 옵션들 필독할 것. 잘 이해 안되는 옵션은 물어보세요**
https://github.com/scyonggg/DNN_22_2/blob/de3d5b7ee8b224c5f3ca8c8c50be233a42632513/train.py#L24-L44

1. `conf/settings.py`에서 DATA_PATH를 train dataset 위치로 수정
2. wandb를 사용할 경우

    a. `script.sh`에 --wandb를 추가
  
    b. 터미널에 wandb login 입력 후, wandb 홈페이지에서 API Keys를 복-붙

    c. `train.py`에서 `wandb.init()` 을 다음과 같이 설정.
    
      - project : 본인 이름
      - entity : "dnn_22_2"
      - name : 본인이 실험할 내용을 짧게 요약 (e.g. CSE_Loss_cosineLR 식으로). `script.sh`에서 run_name을 수정
      - 예시
    
    
    https://github.com/scyonggg/DNN_22_2/blob/47ca6215bf4d6478598df74b362e924b1d1735aa/train.py#L59

3. 0번의 parser에 따라 본인이 실험할 내용에 맞춰 `script.sh` 옵션들 수정 및 확인 후 터미널에 `bash script.sh` 실행
    
    a. `--gpus`에 본인이 사용할 gpu 번호를 넣으면 됨. (수정했음)


4. wandb에 잘 올라오는지 확인


## To DO
1. 도커 업데이트 or 콘다 업데이트
2. requirement 업로드


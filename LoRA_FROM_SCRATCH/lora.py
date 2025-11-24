import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayerBase:
    '''
    LoRA Layer의 공통 속성을 정의하는 부모 클래스

    Attributes:
        rank (int): LoRA의 rank
        lora_alpha (float): LoRA의 alpha
        lora_dropout (float): LoRA의 dropout
    
    
    
    '''
    def __init__(self, rank=8, lora_alpha =8, lora_dropout=0.0, use_rslora=True):
        self.rank = rank
        self.lora_alpha = lora_alpha
        
        # Scaling factor
        # Scaling factor = lora_alph/rank 
        
        if use_rslora:
            # RS-LoRA: alpha / sqrt(rank)
            self.scaling = self.lora_alph/(self.rank ** 0.5)
        else:
            self.scaling = self.lora_alpha/self.rank
            
            
        # 드롭 아웃 설정
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x:x
            

class LoRALinear(nn.Linear, LoRALayerBase):
    """
    LoRA가 적용된 Linear Layer입니다.
    nn.Linear를 상속받았으므로, 기존 Linear Layer처럼 동작하면서 LoRA 기능이 추가됩니다.
    """

    def __init__(self,in_features,out_features,bias=True,rank=8,lora_ahpha=8,lora_dropout=0.0, use_rslora=True,**kwargs):
        """
            in_features: 입력의 차원
            out_features: 출력의 차원
            bias: 편향을 사용할지 여부
            rank: LoRA의 rank
            lora_alpha: LoRA의 alpha
            lora_dropout: LoRA의 dropout
            use_rslora: RS-LoRA를 사용할지 여부
        """
    
        # 1. 부모 클래스 인 nn.Linear와 LoRALayerBase초기화
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        LoRALayerBase.__init__(self, rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, use_rslora=use_rslora)
        """
            **kwargs는 Python의 가변 키워드 인자(Variable Keyword Arguments)입니다!
        """
       
       
        # 2. 원래 Linear Layer의 웨이트는 학습되지 않도록 고정 (Freeze)
        self.weight.requires_grad = False
        """
        self.weight: 원래 Linear Layer의 웨이트
        self.weight.requires_grad의 의미 : 학습 여부를 결정
        False로 설정하면 학습하지 않음
        """
        
        # LoRA 행렬 A와 B 정의하기
        # LoRA는  W + del(W)
        # del(W) = (A @ B) * scaling
        # A와 B는 학습 가능한 파라미터(nn.Parameter)여야 합니다.
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        
        # 논문에 따르면 A는 랜덤 initialization을 B는 0으로 초기화한다
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        """
        kaiming_uniform_ : He init 방법을 사용하여 초기화
        """
        
        
    def forward(self,x):
        
        """
        오버 라이트 전에는
        1. Linear Layer의 결과 계산 (F.linear 사용 또는 super().forward(x))
        
        2. LoRA Branch의 결과 계산 (행렬 곱셈 @ 사용)
        
            - dropout 적용: self.lora_dropout(x)
        - A 곱하기
        - B 곱하기
        - scaling 곱하기
        3. 두 결과 더하기
        """
        
        #1. 원래 linear layer 결과 출력
        original_output = F.linear(x, self.weight, bias=self.bias)
        
        # 2. LoRA 출력 계산
        # 1. x에 dropout 적용
        # 2. A 행렬 곱하기 (@)
        # 3. B 행렬 곱하기 (@)
        # 4. scaling 곱하기 (*) -> 스칼라 값이라서 * 사용 가능
        lora_output = (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        
        return original_output + lora_output
    
    
class LoRAModel(nn.Module):
    """
    기존 모델(예: BERT, Roberta 등)을 감싸서 LoRA를 적용하는 Wrapper 클래스입니다.
    
    """
    def __init__(self, model, rank=8, lora_alpha=8, lora_dropout=0.0, target_modules=["query", "value"]):
        super().__init__()
        self.model = model
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        
        # 1. 기존 모델의 모든 파라미터를 얼립니다.(Freeze like SHINee)
        for param in self.model.parameters():
            param.requires_grad = False
            
        
        # 2. LoRA 레이어 교체 시작
        """
        _ (언더스코어)의 의미: 파이썬에서 함수 이름 앞에 _를 붙이는 건 
        "이건 내부용(Private)이니까 밖에서 함부로 부르지 마세요!"라는 약속
        
        apply_lora(): 밖에서 model.apply_lora() 이렇게 당당하게 써도 됨.
        _apply_lora(): "이 클래스 안에서만 쓸 거니까 건드리지 마." (물론 강제성은 없지만, 개발자들끼리의 매너입니다. :))
        """
        
        self._apply_lora()
        
        #3. 교체된 후, 학습 가능한 파라미터 수 확인 (디버깅용)
        self._print_trainable_parameters()
    


        
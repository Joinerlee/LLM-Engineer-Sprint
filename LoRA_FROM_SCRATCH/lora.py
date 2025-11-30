import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import List, Optional, Union


#dataclass를 사용하면 구조체처럼 쓸수있다! init 부터 이런거
@dataclass
class LoraConfig:
    rank: int = 8
    lora_alpha : float = 8.0
    target_modules: Optional[List[str]] = None
    lora_dropout : float = 0.0
    use_rslora : bool = True
    
    def __post_init__(self):
        """
        LoraConfig 필드의 후처리 초기화.
        Mutable default argument 문제를 우회하여 target_modules의 
        기본값을 ["query", "value"]로 설정합니다.
        """
        if self.target_modules is None:
            self.target_modules = ["query", "value"]

        if self.user_rslora:
            self.scaling = self.lora_alpha/ (self.rank **0.5)
        else:
            self.scaling = self.lora_alpha/ self.rank

class LoRALayerBase:
    '''
    LoRA Layer의 공통 속성을 정의하는 부모 클래스

    Attributes:
        rank (int): LoRA의 rank
        lora_alpha (float): LoRA의 alpha
        lora_dropout (float): LoRA의 dropout
    
    
    '''
    def __init__(self, config : LoraConfig):
        self.rank = config.rank
        self.lora_alpha = config.lora_alpha
        self.use_rslora = config.use_rslora
        
        # Scaling factor
        # Scaling factor = lora_alpha /rank 
        
        if self.use_rslora:
            # RS-LoRA: alpha / sqrt(rank)
            self.scaling = self.lora_alpha/(self.rank ** 0.5)
        else:
            self.scaling = self.lora_alpha/self.rank
            
            
        # 드롭 아웃 설정
        if config.lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=config.lora_dropout)
        else:
            self.lora_dropout = lambda x:x
        
        self.merged =False
            
    def merge_weights(self):
        if self.merged:
            print("이미 병합되었습니다")
            return
        self.merge_implementation()
        self.merged = True
    
    def _merge_implementation(self):
        raise NotImplementedError("서브 클래스에서 구현해야 합니다.")
    
class LoRALinear(nn.Linear, LoRALayerBase):
    """
    LoRA가 적용된 Linear Layer입니다.
    nn.Linear를 상속받았으므로, 기존 Linear Layer처럼 동작하면서 LoRA 기능이 추가됩니다.
    """

    def __init__(self, in_features, out_features, config: LoraConfig, bias=True, **kwargs):
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
        LoRALayerBase.__init__(self, config)
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
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, self.rank))
        self.lora_B = nn.Parameter(torch.zeros(self.rank, out_features))
        
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        # 논문에 따르면 A는 랜덤 initialization을 B는 0으로 초기화한다        
        """
        kaiming_uniform_ : He init 방법을 사용하여 초기화
        """
    
    
    """
    수식: weight += (A @ B).T * scaling
    """
    def _merge_implementation(self):
        self.weight +=(self.lora_A @ self.lora_B).T * self.scaling        
    
    
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
    
    def _apply_lora(self):
        """
        모델의 모든 모듈을 순회하면서,
        target_modules에 해당하는 모듈을 LoRA 레이어로 교체합니다.
        
        1. 모델의 모든 모듈을 순회
        2. 모듈 이름과 타입 확인
        3. target_modules에 해당하면 LoRALinear로 교체
        4. 교체된 모듈을 모델에 다시 할당
        """
        self._replace_modules(self.model)
    
    def _replace_modules(self, module):
        # module의 자식들을 순회합니다
        # 여기서 module은 self.model이 될 수도 있고, 그 하위 모듈이 될 수도 있습니다.
        
        # module의 자식들을 순회합니다.
        for name, child in module.named_children():
            
            # 1. 만약 자식이 또 자식을 가지고 있다면 -> 재귀호출!
            # ps: named_chidren(): 직계 자식만 보기 vs 모든 자손보기 named_modules()
            
            if len(list(child.children())) > 0:
                self._replace_modules(child)
                
            # 2. 자식이 Linear 레이어, 이름이 타겟에포함된다면? -> 교체!!
            if isinstance(child, nn.Linear) and any(t in name for t in self.target_modules):

                """
                LoRA: Low-Rank Adaptation of Large Language Models에 의하면
                target 모듈은 주로 Qurey랑 Value가 제일 좋았다고 한다
                #   - Q만 LoRA 적용
                #   - K만 LoRA 적용
                #   - V만 LoRA 적용
                #   - Q+V LoRA 적용
                #   - 전체 W LoRA 적용
                
                # 실험 결론:
                #   Q+V 조합이 "파라미터 수 대비 성능 향상"이 가장 컸음.
                #   모든 W를 학습시키면 성능은 더 좋지만,
                #   파라미터가 너무 많아져서 LoRA 효율성이 사라짐.
                #   즉, Q+V = 최적의 sweet spot (가성비 최고)
                
                # 나의 질문 Qurey는 사용자입력과 연관되어있는데? : 사실은 어떻게 입력을 해석할지에 핵심이였다!!
                Query : 해석법
                Value : 해석법에 따른 값 사과를 보고 apple일지 apolization일지
                """
                
                #1. 새 LoRALinear를 생성한다
                new_layer = LoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=(child.bias is not None),
                    rank=self.rank,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                )
                
                
                # 2. 가중치 복사 (기존 학습된값 복사)
                new_layer.weight.data = child.weight.data
                if child.bias  is not None:
                    new_layer.bias.data = child.bias.data
                    
                    
                # 3. 핵심 of 핵심 교체를 한다
                """
                기존의 linear layer 였던것이 이제는 LoRALinear로 교체된다
                """
                setattr(module, name, new_layer)
                
                
                print(f"LoRA Layer 적용 완료!")
            
    def forward(self,x):
        return self.model(x)
    
    
    def _print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%")

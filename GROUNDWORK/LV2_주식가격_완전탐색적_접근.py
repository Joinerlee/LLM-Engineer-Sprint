def solution(prices: list):
    answer = []
    list_length = len(prices)

    for i in range(list_length):
        count = 0
        for j in range(i + 1, list_length):
            count += 1
            # 가격이 떨어지는 순간 멈춤
            if prices[j] < prices[i]:
                break
        # 떨어지든 말든, j 루프 끝나고 나서 count를 추가
        answer.append(count)

    return answer

"""
논리를 코드로 옮길 때 검증하는 사고 과정 부족

1. 자연어 → 코드 조건으로 번역하는 정확도
(말은 <인데 손은 > 치고 있는 것)

2. “모든 경우의 수를 커버하고 있는지”를 체크하는 습관
(특히 루프 안/밖, break 타는/안 타는 경우)

3. 인덱스/범위/마지막 원소(off-by-one)에 대한 감각

4. 코드 완성 후, 작은 예제로 직접 시뮬레이션(드라이런)하는 습관
"""
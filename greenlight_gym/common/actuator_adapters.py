# -*- coding: utf-8 -*-
def position_to_time(target_0_1: float, full_open_s: int, full_close_s: int, current_pos_0_1: float):
    """
    개도율 명령을 '열기/닫기/유지 + 구동시간(초)'로 변환.
    장치가 시간 기반으로만 동작하는 경우에 사용.
    """
    target = max(0.0, min(1.0, float(target_0_1)))
    curr = max(0.0, min(1.0, float(current_pos_0_1)))
    if target > curr:
        delta = target - curr
        return ("open",  int(round(delta * full_open_s)))
    elif target < curr:
        delta = curr - target
        return ("close", int(round(delta * full_close_s)))
    return ("hold", 0)



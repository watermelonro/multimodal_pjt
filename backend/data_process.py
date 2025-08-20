import json
from typing import List, Dict, Any
import statistics
import os

def analyze_concentration_changes(data: List[Dict]) -> Dict[str, Any]:
    """
    집중도 변화점을 중심으로 학습 데이터를 분석하는 함수
    """

    

    # 잡음 매핑
    noise_labels = {
        0: "없음",
        1: "무음",
        2: "자동차",
        3: "사이렌",
        4: "키보드",
        5: "대화소리",
    }

    # 전체 평균 계산
    total_concentration = []
    total_yaw = []
    total_pitch = []
    noise_counts = {}

    for entry in data:
        # 집중도 점수 (원본 데이터 사용)
        conc_score = entry["result"]["num"]
        total_concentration.append(conc_score)

        # 시선 데이터
        total_yaw.append(
            abs(entry["pose"]["yaw"])
        )  # 절댓값으로 변환 (좌우 움직임 정도)
        total_pitch.append(
            abs(entry["pose"]["pitch"])
        )  # 절댓값으로 변환 (상하 움직임 정도)

        # 잡음 카운트
        noise_num = entry["noise"]["num"]
        noise_label = noise_labels.get(noise_num, "알수없음")
        noise_counts[noise_label] = noise_counts.get(noise_label, 0) + 1

    # 전체 평균 계산
    avg_concentration = round(statistics.mean(total_concentration), 2)
    avg_yaw = round(statistics.mean(total_yaw), 2)
    avg_pitch = round(statistics.mean(total_pitch), 2)
    dominant_noise = max(noise_counts, key=noise_counts.get)

    # 집중도 변화점 찾기 (2점 이상 변화 또는 집중/비집중 상태 전환)
    change_points = []
    previous_score = total_concentration[0]
    current_segment_start = 0

    for i, entry in enumerate(data[1:], 1):  # 두 번째 항목부터 시작
        current_score = total_concentration[i]

        # 집중도가 크게 변했는지 판단
        score_change = abs(current_score - previous_score)
        concentration_state_change = is_concentration_state_changed(
            previous_score, current_score
        )

        if score_change >= 2 or concentration_state_change:
            # 이전 구간 정리
            if i > current_segment_start:
                segment_data = data[current_segment_start:i]
                segment_concentration = total_concentration[current_segment_start:i]
                segment_yaw = total_yaw[current_segment_start:i]
                segment_pitch = total_pitch[current_segment_start:i]

                change_points.append(
                    {
                        "time_range": f"{segment_data[0]['timestamp']['start']}-{segment_data[-1]['timestamp']['end']}",
                        "duration_seconds": segment_data[-1]["timestamp"]["end"]
                        - segment_data[0]["timestamp"]["start"],
                        "avg_concentration_score": round(
                            statistics.mean(segment_concentration), 2
                        ),
                        "concentration_state": get_concentration_state(
                            statistics.mean(segment_concentration)
                        ),
                        "is_focused": is_focused_state(
                            statistics.mean(segment_concentration)
                        ),
                        "avg_head_movement": {
                            "yaw": round(statistics.mean(segment_yaw), 2),
                            "pitch": round(statistics.mean(segment_pitch), 2),
                        },
                        "head_stability": get_head_stability(
                            statistics.mean(segment_yaw), statistics.mean(segment_pitch)
                        ),
                        "dominant_noise": get_dominant_noise_in_segment(segment_data),
                        "change_type": get_change_type(previous_score, current_score),
                    }
                )

            current_segment_start = i

        previous_score = current_score

    # 마지막 구간 처리
    if current_segment_start < len(data):
        segment_data = data[current_segment_start:]
        segment_concentration = total_concentration[current_segment_start:]
        segment_yaw = total_yaw[current_segment_start:]
        segment_pitch = total_pitch[current_segment_start:]

        change_points.append(
            {
                "time_range": f"{segment_data[0]['timestamp']['start']}-{segment_data[-1]['timestamp']['end']}",
                "duration_seconds": segment_data[-1]["timestamp"]["end"]
                - segment_data[0]["timestamp"]["start"],
                "avg_concentration_score": round(
                    statistics.mean(segment_concentration), 2
                ),
                "concentration_state": get_concentration_state(
                    statistics.mean(segment_concentration)
                ),
                "is_focused": is_focused_state(statistics.mean(segment_concentration)),
                "avg_head_movement": {
                    "yaw": round(statistics.mean(segment_yaw), 2),
                    "pitch": round(statistics.mean(segment_pitch), 2),
                },
                "head_stability": get_head_stability(
                    statistics.mean(segment_yaw), statistics.mean(segment_pitch)
                ),
                "dominant_noise": get_dominant_noise_in_segment(segment_data),
                "change_type": "final_segment",
            }
        )

    # 결과 정리
    result = {
        "session_overview": {
            "total_duration_seconds": data[-1]["timestamp"]["end"]
            - data[0]["timestamp"]["start"],
            "total_duration_minutes": round(
                (data[-1]["timestamp"]["end"] - data[0]["timestamp"]["start"]) / 60, 1
            ),
            "avg_concentration_score": avg_concentration,
            "avg_concentration_state": get_concentration_state(avg_concentration),
            "overall_focus_status": is_focused_state(avg_concentration),
            "avg_head_movement": {"yaw": avg_yaw, "pitch": avg_pitch},
            "head_stability": get_head_stability(avg_yaw, avg_pitch),
            "dominant_noise": dominant_noise,
            "total_segments": len(change_points),
            "focus_ratio": calculate_focus_ratio(change_points),
        },
        "concentration_segments": change_points,
        "learning_insights": generate_insights(
            change_points, avg_concentration, dominant_noise
        ),
    }

    return result

def is_concentration_state_changed(prev_score: int, curr_score: int) -> bool:
    """집중 상태가 변했는지 판단 (집중함 <-> 집중하지 않음)"""
    prev_focused = prev_score <= 1  # 0, 1: 집중함
    curr_focused = curr_score <= 1
    return prev_focused != curr_focused


def get_concentration_state(score: float) -> str:
    """집중도 점수를 상태 문자열로 변환"""
    if score >= 3.5:
        return "졸음"
    elif score >= 2.5:
        return "지루함"
    elif score >= 1.5:
        return "차분함(집중하지 않음)"
    elif score >= 0.5:
        return "차분함(집중함)"
    else:
        return "흥미로움(집중함)"


def is_focused_state(score: float) -> bool:
    """집중 상태인지 판단"""
    return score <= 1.5  # 0, 1: 집중함


def get_head_stability(avg_yaw: float, avg_pitch: float) -> str:
    """머리 움직임 안정성 평가"""
    movement_score = (avg_yaw + avg_pitch) / 2
    if movement_score < 20:
        return "매우 안정"
    elif movement_score < 30:
        return "안정"
    elif movement_score < 40:
        return "보통"
    else:
        return "불안정"


def get_change_type(prev_score: int, curr_score: int) -> str:
    """변화 유형 분류"""
    if curr_score < prev_score:
        return "집중도 향상"
    elif curr_score > prev_score:
        return "집중도 저하"
    else:
        return "유지"


def get_dominant_noise_in_segment(segment_data: List[Dict]) -> str:
    """구간 내 주요 소음 타입 찾기"""
    noise_labels = {
        0: "없음",
        1: "무음",
        2: "자동차",
        3: "사이렌",
        4: "키보드",
        5: "대화소리",
    }
    noise_counts = {}

    for entry in segment_data:
        noise_num = entry["noise"]["num"]
        noise_label = noise_labels.get(noise_num, "알수없음")
        noise_counts[noise_label] = noise_counts.get(noise_label, 0) + 1

    return max(noise_counts, key=noise_counts.get)


def calculate_focus_ratio(segments: List[Dict]) -> float:
    """전체 세션에서 집중한 시간 비율 계산"""
    total_time = sum(seg["duration_seconds"] for seg in segments)
    focused_time = sum(seg["duration_seconds"] for seg in segments if seg["is_focused"])
    return round(focused_time / total_time * 100, 1) if total_time > 0 else 0.0


def generate_insights(
    segments: List[Dict], avg_concentration: float, dominant_noise: str
) -> List[str]:
    """학습 패턴 인사이트 생성"""
    insights = []

    if len(segments) == 0:
        return ["데이터가 부족합니다."]

    # 전체 집중도 평가
    if avg_concentration <= 1:
        insights.append("전반적으로 집중을 잘 유지했습니다! 👍")
    elif avg_concentration <= 2:
        insights.append("대체로 집중하는 모습을 보였지만 개선의 여지가 있습니다.")
    else:
        insights.append(
            "집중도가 전반적으로 낮았습니다. 학습 환경이나 방법을 점검해보세요."
        )

    # 초반 vs 후반 비교
    if len(segments) >= 2:
        first_half = segments[: len(segments) // 2]
        second_half = segments[len(segments) // 2 :]

        first_avg = statistics.mean(
            [seg["avg_concentration_score"] for seg in first_half]
        )
        second_avg = statistics.mean(
            [seg["avg_concentration_score"] for seg in second_half]
        )

        if first_avg < second_avg:  # 점수가 낮을수록 집중도 높음
            insights.append(
                "학습 초반에 더 집중했습니다. 시간이 지날수록 집중도가 떨어지는 경향이 있어요."
            )
        elif first_avg > second_avg:
            insights.append("시간이 지날수록 집중도가 향상되었습니다! 좋은 패턴이에요.")

    # 시선 안정성 분석
    unstable_segments = [
        seg for seg in segments if seg["head_stability"] in ["불안정", "보통"]
    ]
    if len(unstable_segments) > len(segments) / 2:
        insights.append(
            "시선이 자주 움직였습니다. 한 곳에 집중하는 연습이 필요해 보여요."
        )

    # 소음 영향 분석
    if dominant_noise == "사이렌":
        insights.append("사이렌 소리에 특히 민감하게 반응하는 것 같습니다.")
    elif dominant_noise == "대화소리":
        insights.append(
            "대화 소리가 집중에 방해가 되는 것 같습니다. 조용한 환경에서 학습해보세요."
        )
    elif dominant_noise == "자동차":
        insights.append("자동차 소리가 학습에 영향을 주고 있습니다.")

    # 집중 구간 분석
    focused_segments = [seg for seg in segments if seg["is_focused"]]
    if len(focused_segments) > 0:
        avg_focused_duration = statistics.mean(
            [seg["duration_seconds"] for seg in focused_segments]
        )
        if avg_focused_duration > 180:  # 3분 이상
            insights.append("집중할 때는 오랫동안 지속하는 좋은 패턴을 보입니다.")
        else:
            insights.append("집중 지속 시간이 짧습니다. 점진적으로 늘려나가 보세요.")

    return insights


# 사용 예시
if __name__ == "__main__":
    # 실제 데이터를 여기에 넣으세요
    result_path = "result_경민서.json"
    data = os.path.join("analysis", result_path)
    with open(data, "r", encoding="utf-8") as f:
        sample_data = json.load(f)
    sample_data = sorted(sample_data, key=lambda x: x["timestamp"]["start"])

    # 분석 실행
    analysis_result = analyze_concentration_changes(sample_data)

    # 결과 출력 (예쁘게 포맷팅)
    print("=== 학습 세션 분석 결과 ===")
    print(
        f"총 학습 시간: {analysis_result['session_overview']['total_duration_minutes']}분"
    )
    print(
        f"평균 집중도: {analysis_result['session_overview']['avg_concentration_state']}"
    )
    print(f"집중 시간 비율: {analysis_result['session_overview']['focus_ratio']}%")
    print(f"주요 소음: {analysis_result['session_overview']['dominant_noise']}")
    print()
    print("=== 학습 인사이트 ===")
    for insight in analysis_result["learning_insights"]:
        print(f"• {insight}")
    print()
    print("=== 상세 분석 데이터 ===")
    print(json.dumps(analysis_result, indent=2, ensure_ascii=False))




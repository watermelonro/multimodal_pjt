import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os
import platform
import io

# 💡 운영체제에 맞는 한글 폰트 설정
system_name = platform.system()
if system_name == "Darwin":  # macOS
    plt.rc("font", family="AppleGothic")
elif system_name == "Windows":  # Windows
    plt.rc("font", family="Malgun Gothic")
# 다른 운영체제에 대한 폰트 설정이 필요하다면 여기에 추가

plt.rcParams["axes.unicode_minus"] = False


def sec_to_time(sec):
    """초 단위를 시:분:초 형식으로 변환"""
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = int(sec % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def sec_to_time_str(sec):
    """초 단위를 시:분:초 형식으로 변환"""
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = int(sec % 60)
    if hours != 0:
        return f"{hours}시간 {minutes}분 {seconds}초"
    else:
        return f"{minutes}분 {seconds}초"


def plot_feedback_and_df(result, user_name):
    """학습 몰입도 분석 결과를 시각화하고, 이미지 바이트를 반환하는 함수"""

    # 📊 그래프 크기 설정
    plt.figure(figsize=(12, 6))

    # 데이터 준비
    x = [r["timestamp"]["start"] for r in result]
    y = [-r["result"]["num"] for r in result]

    # 변화 지점만 추출
    change_points = [x[0]]
    up_down = [1]  # up: 1, down: -1
    for i in range(1, len(y)):
        if y[i] != y[i - 1]:
            change_points.append(x[i])
            if y[i] > y[i - 1]:
                up_down.append(1)
            else:
                up_down.append(-1)
    change_points.append(x[-1])
    up_down.append(1)
    labels = [sec_to_time(point) for point in change_points]

    # 📈 그래프 그리기
    plt.plot(
        x,
        y,
        linestyle="-",
        color="#4CAF50",
        linewidth=2,
        marker="o",
        markersize=5,
        markerfacecolor="white",
    )

    # y축 설정
    plt.yticks(
        ticks=[-4, -3, -2, -1, 0],
        labels=[
            "졸음",
            "지루함\n(집중↓)",
            "차분함\n(집중↓)",
            "차분함\n(집중↑)",
            "흥미로움\n(집중↑)",
        ],
        fontsize=10,
    )

    # 🎯 변화 지점 레이블 + 선 연결
    ax = plt.gca()

    for i, (point, num) in enumerate(zip(change_points, up_down)):
        label = labels[i]
        y_val = y[x.index(point)]

        # 🔍 간격에 따라 위치 조정
        text_y = y_val + num * 0.4
        arrow = dict(arrowstyle="-", color="gray", linewidth=0.8)

        # 📍 레이블 및 연결선
        ax.annotate(
            label,
            xy=(point, y_val),
            xytext=(point, text_y),
            textcoords="data",
            ha="center",
            fontsize=9,
            arrowprops=arrow,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.5),
        )

    # ❌ 기본 xticks 비우기
    plt.xticks([])

    # 레이블 및 타이틀
    plt.xlabel("시간 (시:분:초)", fontsize=12)
    plt.ylabel("몰입 상태", fontsize=12)
    plt.title(f"{user_name} 학습 몰입도 분석", fontsize=16, weight="bold", pad=40)

    # 🎨 그래프 미화
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#fdfdfd")  # 밝은 배경

    plt.tight_layout()

    # 그래프를 메모리 버퍼에 저장
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", dpi=300)
    img_buffer.seek(0)
    image_bytes = img_buffer.read()
    plt.close()  # 메모리 누수 방지를 위해 그래프 객체 닫기

    processed_data = []
    stt_chunk = []
    temp_str = ""

    def state_define(state_num):
        if state_num <= 1:
            state = 1
        else:
            state = -1
        return state

    def change_state(state):
        if state == 1:
            state_str = "집중"
        else:
            state_str = "집중 못함"
        return state_str

    prev_state = state_define(result[0]["result"]["num"])
    for r in result:
        processed_item = {
            "start_time": sec_to_time_str(r["timestamp"]["start"]),
            "concentration": r["result"]["str"],
            "pose": {"yaw": int(r["pose"]["yaw"]), "pitch": int(r["pose"]["pitch"])},
            "noise": r["noise"]["str"],
        }
        current_state = state_define(r["result"]["num"])
        if current_state == prev_state:
            temp_str += r["text"]
        else:
            stt_chunk.append([temp_str, change_state(prev_state)])
            prev_state = current_state
            temp_str = r["text"]
        processed_data.append(processed_item)

    stt_chunk.append([temp_str, change_state(prev_state)])

    data_str = json.dumps(processed_data, ensure_ascii=False, separators=( ",", ":"))

    return data_str, stt_chunk, image_bytes


if __name__ == "__main__":
    result_path = "result_경민서_modified.json"
    data = os.path.join("analysis", result_path)
    with open(data, "r", encoding="utf-8") as f:
        result = json.load(f)
    result = sorted(result, key=lambda x: x["timestamp"]["start"])
    # 테스트 코드도 새로운 반환값에 맞게 수정
    data_str, stt_chunk, _ = plot_feedback_and_df(result, "경민서")
    print([chunk[1] for chunk in stt_chunk])
    print(len(stt_chunk))

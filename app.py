import streamlit as st
import requests
import time
import pandas as pd
import re

st.set_page_config(
    page_title="인바디 기반 맞춤 운동·식단 추천",
    page_icon="🏋️‍♂️",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton > button {
        background: linear-gradient(90deg, #35a2e8 0%, #2cc993 100%);
        color: white;
        font-weight: 600;
        border-radius: 1.5rem;
        padding: 0.6em 2.5em;
        font-size: 1.1em;
        margin-top: 1.5em;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("## 🏋️‍♂️ 인바디 기반 1주일 운동·식단 추천")
st.write(
    "인바디 결과 사진을 업로드하면, 목표에 맞춘 일주일 운동 루틴과 식단을 AI가 직접 추천해드립니다. "
    "식단/운동 계획은 신뢰도 높은 문서 근거 기반으로 생성됩니다."
)

with st.expander("🙋‍♂️ 사용법 안내"):
    st.markdown(
        """
        1. **운동 목표**를 선택하세요.<br>
        2. **인바디 결과 사진**을 업로드하세요.<br>
        3. **[추천받기]** 버튼을 누르면, 결과가 아래에 생성됩니다.<br>
        - (예시 인바디 이미지는 [여기](https://www.inbody.com/kr/board/view.asp?idx=81) 참고)
        """,
        unsafe_allow_html=True,
    )

col1, col2 = st.columns([1, 2])
with col1:
    goal = st.selectbox("운동 목표", ["감량", "벌크업", "유지"], index=0)
with col2:
    image_file = st.file_uploader("인바디 결과 사진 업로드", type=["png", "jpg", "jpeg"])

btn = st.button("🎁 추천받기", use_container_width=True)

if btn and image_file:
    files = {"image": (image_file.name, image_file, image_file.type)}
    data = {"goal": goal}
    with st.spinner("AI가 인바디를 분석 중입니다... ⏳"):
        start_time = time.time()
        try:
            resp = requests.post("http://172.19.1.62:3001/recommend", data=data, files=files, timeout=180)
            end_time = time.time() - start_time
            if resp.status_code == 200:
                result = resp.json()
                st.success(f"📚 추천이 완료되었습니다! (소요 시간: {end_time:.2f}초)")
                st.markdown("---")

                # 인바디 표/운동표/식단표 자동 분리
                result_text = result["result_text"]
                sections = re.split(r"(### [^\n]+)", result_text)
                section_dict = {}
                current = ""
                for s in sections:
                    if s.startswith("###"):
                        current = s
                        section_dict[current] = ""
                    elif current:
                        section_dict[current] += s

                for title, content in section_dict.items():
                    st.markdown(f"#### {title.replace('### ', '').strip()}")
                    # 마크다운 표는 판다스 표로 예쁘게
                    if "|" in content and "---" in content:
                        try:
                            df = pd.read_csv(
                                pd.compat.StringIO(content.strip()), sep="|", engine="python"
                            )
                            st.dataframe(df)
                        except:
                            st.markdown(content)
                    else:
                        st.markdown(content)
                    st.markdown("---")
            else:
                st.error(f"❗️추천 실패: {resp.text}")
        except Exception as e:
            st.error(f"❗️서버 요청 중 오류 발생: {str(e)}")

elif btn and not image_file:
    st.warning("📸 인바디 결과 이미지를 먼저 업로드해 주세요.")

st.markdown(
    """
    <hr>
    <p style="text-align: center; color: #999; font-size: 0.9em;">
    ⓒ 2025 인바디 AI 추천 시스템 DEMO - Powered by GPT-4o, LangChain, Qdrant
    </p>
    """,
    unsafe_allow_html=True,
)

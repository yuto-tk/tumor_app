import streamlit as st

def result_card(patient_id, res, prob_percent, positive):

    color = "#ff4b4b" if positive else "#4b8bff"

    html = f"""
<div style="background:#222; padding:16px; margin-bottom:16px; border-radius:12px;">

  <div style="font-size:20px; color:white; font-weight:600;">
    {patient_id}
  </div>

  <div style="font-size:17px; margin-top:6px; color:{color};">
    {res}
  </div>

  <div style="font-size:15px; margin-top:4px; color:white;">
    確率: {prob_percent:.1f}%
  </div>

  <div style="height:12px; background:#444; border-radius:6px; margin-top:10px;">
    <div style="height:12px; width:{prob_percent}%; background:{color}; border-radius:6px;"></div>
  </div>

</div>
"""

    st.markdown(html, unsafe_allow_html=True)

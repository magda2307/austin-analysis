import re

with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace st.columns(x) with st.columns(x, gap="large")
text = re.sub(r'st\.columns\((\d+)\)', r'st.columns(\1, gap="large")', text)

# Replace st.markdown("---") with st.divider()
text = text.replace('st.markdown("---")', 'st.divider()')
text = text.replace("st.markdown('---')", 'st.divider()')

with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(text)

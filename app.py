import streamlit as st
import requests

st.set_page_config(page_title="🇰🇪 Kenya Constitution Q&A", page_icon="📜")

st.title("📘 Ask the Constitution of Kenya (2010)")

st.markdown("""
Type a question below to search the Constitution of Kenya 2010.

Examples:
- What are the rights of arrested persons?
- What does the Constitution say about education?
- What is the structure of government in Kenya?
""")

query = st.text_input("🔎 Enter your question")

if st.button("Ask") and query:
    with st.spinner("Looking up the Constitution..."):
        try:
            response = requests.post("http://localhost:8000/ask", json={"query": query})
            data = response.json()

            st.markdown("### 🧠 Answer")
            st.success(data.get("answer", "No answer found."))

            st.markdown("---")
            st.markdown("### 📚 Sources")
            sources = data.get("sources", [])
            if sources:
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**{i}.** `{src}`")
            else:
                st.write("No source metadata available.")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

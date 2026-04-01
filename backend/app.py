import streamlit as st
from rag_pipeline import index_channel, query_channel

st.title("🎥 YouTube Channel RAG")
st.caption("Ask anything across the entire channel")

channel_url = st.text_input(
    "Channel URL",
    placeholder="https://www.youtube.com/@channelname/videos"
)

if st.button("Index Channel") and channel_url:
    with st.spinner("Indexing channel... this takes a few minutes"):
        index_channel(channel_url)
    st.success("Channel indexed! Start asking questions.")

question = st.text_input("Your question")
if st.button("Ask") and question:
    with st.spinner("Searching..."):
        result = query_channel(question)
    st.write("**Answer:**", result["answer"])
    st.write("**Sources:**")
    for url in result["sources"]:
        st.markdown(f"- [{url}]({url})")
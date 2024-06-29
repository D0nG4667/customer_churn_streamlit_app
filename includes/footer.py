import streamlit as st

footer = """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f8f9fa;
        color: #333;
        text-align: center;
        padding: 10px;
        box-shadow: 0 -1px 5px rgba(0, 0, 0, 0.1);
    }
    </style>
    <div class="footer">
                &copy; 2024. Made with 💖 <a href="https://www.linkedin.com/in/dr-gabriel-okundaye" target="_blank" style="text-decoration: none;">Gabriel Okundaye</a>
        <span style="color: #aaaaaa;">& Light ✨</span><br>
    </div>
    """

st.markdown(footer, unsafe_allow_html=True)

import streamlit as st
from firebase_config import auth
st.set_page_config(
    page_title="SenseChurn",
    page_icon="📡",
    layout="centered"
)
st.markdown("""

<style>
@import url('https://fonts.googleapis.com/css2?family=Audiowide&display=swap');


h1 {
    color: #818cf8;
    font-family: 'Audiowide', sans-serif !important;
    letter-spacing: 2px;
}
/* Hide Press Enter to apply - all methods */
[data-testid="InputInstructions"],
.st-emotion-cache-1rsyhoq,
small {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}
</style>
""", unsafe_allow_html=True)

def show_login_page():
    st.markdown("<h1 style='white-space: nowrap; font-size: 2.5rem;'> Welcome to SenseChurn</h1>", unsafe_allow_html=True)
    st.markdown("Please login or create an account to continue")
    st.divider()

    page = st.radio("", ["🔐 Login", "📝 Sign Up"], horizontal=True)
    st.divider()

    if page == "🔐 Login":
        st.subheader("Login")
        email    = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", use_container_width=True, key="login_btn"):
            if email == "" or password == "":
                st.error("❌ Please enter email and password!")
            else:
                try:
                    user = auth.sign_in_with_email_and_password(email, password)
                    if user:
                        st.session_state["user"] = user
                        st.session_state["email"] = email
                        st.rerun()
                except Exception as e:
                    error_message = str(e)
                    if "INVALID_PASSWORD" in error_message or "EMAIL_NOT_FOUND" in error_message:
                        st.error("❌ Invalid email or password!")
                    elif "TOO_MANY_ATTEMPTS" in error_message:
                        st.error("❌ Too many attempts! Try again later.")
                    else:
                        st.error("❌ Invalid email or password!")
    elif page == "📝 Sign Up":
        st.subheader("Create Account")
        new_email    = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password (min 6 characters)",
                                      type="password", key="signup_password")
        confirm      = st.text_input("Confirm Password",
                                      type="password", key="confirm_password")

        if st.button("Create Account", use_container_width=True, key="signup_btn"):
            if new_email == "" or new_password == "":
                st.error("❌ Please fill all fields!")
            elif new_password != confirm:
                st.error("❌ Passwords don't match!")
            elif len(new_password) < 6:
                st.error("❌ Password must be at least 6 characters!")
            else:
                try:
                    auth.create_user_with_email_and_password(new_email, new_password)
                    st.success("✅ Account created! Please login.")
                except:
                    st.error("❌ Email already exists or invalid!")
import streamlit as st
from firebase_config import auth

def show_login_page():
    st.title("👋 Welcome to ChurnSight")
    st.markdown("Please login or create an account to continue")
    st.divider()

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])

    with tab1:
        st.subheader("Login")
        email    = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", use_container_width=True):
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state["user"] = user
                st.session_state["email"] = email
                st.success("✅ Logged in successfully!")
                st.rerun()
            except:
                st.error("❌ Invalid email or password!")

    with tab2:
        st.subheader("Create Account")
        new_email    = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password (min 6 characters)", 
                                      type="password", key="signup_password")
        confirm      = st.text_input("Confirm Password", 
                                      type="password", key="confirm_password")

        if st.button("Create Account", use_container_width=True):
            if new_password != confirm:
                st.error("❌ Passwords don't match!")
            elif len(new_password) < 6:
                st.error("❌ Password must be at least 6 characters!")
            else:
                try:
                    auth.create_user_with_email_and_password(new_email, new_password)
                    st.success("✅ Account created! Please login.")
                except:
                    st.error("❌ Email already exists or invalid!")
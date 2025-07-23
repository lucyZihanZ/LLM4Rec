import streamlit as st

# 这里假设你有如下函数（从你的RAG/ML后端导入）
# from backend import get_recommendations

def get_recommendations(user_query):
    # 这里举例，实际替换为你后端推荐函数
    return [
        {"name": "Energy Booster Coffee", "desc": "Rich coffee, boosts energy."},
        {"name": "Immunity Support Smoothie", "desc": "Packed with vitamin C."},
        {"name": "Protein Power Bar", "desc": "High protein for muscle recovery."}
    ], "Augmented info: These products are selected based on your preferences."

st.title("Product Recommendation Demo (RAG + ML)")
st.write("Enter your needs or questions and get personalized product suggestions!")

# 用户输入
user_query = st.text_input("What are you looking for?", "")

if st.button("Recommend"):
    if user_query.strip() == "":
        st.warning("Please enter a query!")
    else:
        recs, aug_info = get_recommendations(user_query)
        st.subheader("Recommended Products")
        for rec in recs:
            st.write(f"**{rec['name']}**: {rec['desc']}")
        st.subheader("Augmented Info (RAG)")
        st.write(aug_info)

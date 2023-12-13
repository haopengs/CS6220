import streamlit as st
import display_crime_map
import density_prediction
import crime_probability_pediction_combined


# 初始化或获取当前页面状态
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'main'

# 根据当前页面状态显示内容
if st.session_state['current_page'] == 'main':
    # 使用 Markdown 来居中标题
    st.markdown("<h1 style='text-align: center;'>Crime Analysis Dashboard</h1>",
                unsafe_allow_html=True)

    # 用 columns 来创建三个列，这里我们使用了空白列来作为填充
    col1, col2, col3, col4, col5 = st.columns([1, 2, 1.5, 2, 1])

    with col2:
        st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
        if st.button('Area Safety   Prediction'):
            st.session_state['current_page'] = 'density_prediction'
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
        if st.button('Display Crime Map'):
            st.session_state['current_page'] = 'display_crime_map'
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
        if st.button('Crime Probability Prediction'):
            st.session_state['current_page'] = 'crime_probability_prediction'
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# 执行子页面的 run 函数
elif st.session_state['current_page'] == 'density_prediction':
    density_prediction.run()
elif st.session_state['current_page'] == 'display_crime_map':
    display_crime_map.run()
elif st.session_state['current_page'] == 'crime_probability_prediction':
    crime_probability_pediction_combined.run()

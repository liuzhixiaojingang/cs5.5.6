import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 加载模型
@st.cache_resource
def load_model():
    model_path = "best_mlp_model.pkl"
    return joblib.load(model_path)

# 主函数
def main():
    st.title("烧伤程度识别系统")
    st.write("请上传CSV文件进行烧伤程度预测 (0-5)")

    # 文件上传
    uploaded_file = st.file_uploader("选择CSV文件", type="csv")
    
    if uploaded_file is not None:
        try:
            # 读取CSV文件
            df = pd.read_csv(uploaded_file)
            
            # 检查列数是否足够
            if len(df.columns) < 21:
                st.error("CSV文件需要至少21列")
                return
                
            # 获取特征数据 (第二行，第2到21列)
            features = df.iloc[1, 1:21].values.astype(float)
            
            # 显示样本名和特征
            st.write(f"样本名: {df.iloc[1, 0]}")
            st.write("特征值:")
            st.write(pd.DataFrame(features.reshape(1, -1), 
                                columns=[f"DL{i+1}" for i in range(20)]))
            
            # 预测按钮
            if st.button("预测"):
                model = load_model()
                prediction = model.predict(features.reshape(1, -1))
                st.success(f"预测结果: 烧伤程度 {prediction[0]}")
                
        except Exception as e:
            st.error(f"处理文件时出错: {e}")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import joblib

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load("./best_mlp_model.pkl")

# 主程序
def main():
    st.title("烧伤识别预测系统")
    model = load_model()

    # 文件上传组件
    uploaded_file = st.file_uploader("上传CSV文件", type="csv")
    
    if uploaded_file is not None:
        try:
            # 读取数据
            df = pd.read_csv(uploaded_file, index_col=0)
            
            # 验证数据格式
            if df.shape[1] != 20 or not all(df.columns == [f"DL_{i+1}" for i in range(20)]):
                st.error("文件格式错误：必须包含DL_1到DL_20共20个特征列，且第一列为样本名")
                return
                
            # 执行预测
            predictions = model.predict(df.values)
            
            # 显示结果
            st.subheader("预测结果")
            result_df = pd.DataFrame({
                "样本名": df.index,
                "预测类别": predictions
            })
            st.dataframe(result_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"处理文件时发生错误: {str(e)}")

if __name__ == "__main__":
    main()
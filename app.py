import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
from collections import Counter
from io import StringIO

st.set_page_config(
    page_title="Proyecto 2 - Reddit Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #111111;
        font-family: "Helvetica", "Segoe UI", sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    [data-testid="stSidebar"] {
        background: #f5f5f5;
        color: #111111;
        border-right: 1px solid #dddddd;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ff4500;
    }
    h1, h2, h3 {
        color: #ff4500;
        font-weight: 700;
    }
    hr {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin: 1rem 0 2rem 0;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #ff4500;
        color: #ffffff;
        border-radius: 999px;
        border: none;
        padding: 0.45rem 1.3rem;
        font-weight: 600;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #ff6a21;
        color: #ffffff;
    }
    textarea, input, .stTextInput>div>div>input {
        background-color: #ffffff !important;
        color: #111111 !important;
        border: 1px solid #cccccc !important;
    }
    .metric-subtitle {
        color: #ff8c42;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Navegación")
pagina = st.sidebar.radio(
    "Ir a:",
    ["Inicio", "Análisis Exploratorio", "Comparación de Modelos", "Predicción de Comentarios"]
)

if pagina == "Inicio":
    st.title("🧩 Proyecto 2 - Clasificación de Violaciones en Comentarios de Reddit")
    st.markdown(
        """
        **Autores:** Daniel Machic (22118) y María José Ramírez (221051)
        **Curso:** Data Science - 2025

        ---
        ### 🎯 Descripción del proyecto
        Clasificación de comentarios del dataset de Jigsaw para identificar violaciones de reglas
        en comunidades de Reddit.

        ---
        ### 🧠 Modelos utilizados
        - LogisticRegression_TFIDF
        - LinearSVC_TFIDF
        - MultinomialNB_TFIDF
        - BiLSTM_Keras

        El modelo final utilizado en la aplicación es **LinearSVC_TFIDF** por su equilibrio entre desempeño y costo computacional.
        """
    )

elif pagina == "Análisis Exploratorio":
    st.title("📊 Análisis Exploratorio de Datos")

    try:
        train = pd.read_csv("train.csv")

        st.subheader("Vista previa del dataset")
        st.dataframe(train.head(), use_container_width=True)

        st.markdown('<p class="metric-subtitle">Distribución de la variable objetivo</p>', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.countplot(data=train, x="rule_violation", ax=ax1, palette=["#ff4500", "#999999"])
        ax1.set_title("Distribución de violaciones de reglas")
        ax1.set_xlabel("Violación de reglas (0 = No, 1 = Sí)")
        ax1.set_ylabel("Cantidad de comentarios")
        st.pyplot(fig1)

        st.markdown('<p class="metric-subtitle">Longitud de los comentarios</p>', unsafe_allow_html=True)
        train["comment_length"] = train["body"].astype(str).apply(len)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(train["comment_length"], bins=50, kde=True, ax=ax2, color="#ff4500")
        ax2.set_title("Distribución de la longitud de los comentarios")
        ax2.set_xlabel("Número de caracteres")
        ax2.set_ylabel("Frecuencia")
        st.pyplot(fig2)

        st.markdown('<p class="metric-subtitle">Palabras más frecuentes 🧠</p>', unsafe_allow_html=True)
        text_all = " ".join(train["body"].astype(str)).lower()
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text_all)
        common_words = Counter(tokens).most_common(20)

        if common_words:
            freq_df = pd.DataFrame(common_words, columns=["Palabra", "Frecuencia"])
            fig3, ax3 = plt.subplots(figsize=(7, 6))
            sns.barplot(data=freq_df, x="Frecuencia", y="Palabra", ax=ax3, color="#ff8c42")
            ax3.set_title("Top 20 palabras más frecuentes")
            ax3.set_xlabel("Frecuencia")
            ax3.set_ylabel("Palabra")
            st.pyplot(fig3)
        else:
            st.info("No se encontraron palabras frecuentes.")

    except Exception as e:
        st.error(f"Error al cargar datos: {e}")

elif pagina == "Comparación de Modelos":
    st.title("📈 Comparación de Modelos")

    data_modelos = [
        {"Modelo": "LinearSVC_TFIDF",        "Accuracy": 0.739, "Precision": 0.698, "Recall": 0.854, "F1": 0.769, "Tiempo (s)": 2.30},
        {"Modelo": "LogisticRegression_TFIDF","Accuracy": 0.732, "Precision": 0.690, "Recall": 0.854, "F1": 0.764, "Tiempo (s)": 5.81},
        {"Modelo": "BiLSTM_Keras",           "Accuracy": 0.756, "Precision": 0.756, "Recall": 0.767, "F1": 0.761, "Tiempo (s)": 46.33},
        {"Modelo": "MultinomialNB_TFIDF",    "Accuracy": 0.704, "Precision": 0.652, "Recall": 0.893, "F1": 0.754, "Tiempo (s)": 1.25},
    ]

    df_modelos = pd.DataFrame(data_modelos)
    st.subheader("Tabla de métricas")
    st.dataframe(df_modelos, use_container_width=True)

    mostrar_metricas = st.checkbox("Mostrar gráficas de rendimiento y tiempo ⬇️", value=True)

    if mostrar_metricas:
        st.markdown('<p class="metric-subtitle">Rendimiento por métrica</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        df_modelos.set_index("Modelo")[["Accuracy", "Precision", "Recall", "F1"]].plot(
            kind="bar",
            ax=ax,
            color=["#ff4500", "#ff8c42", "#999999", "#555555"]
        )
        plt.xticks(rotation=15, ha="right")
        ax.set_title("Rendimiento de los modelos")
        ax.set_xlabel("")
        ax.set_ylabel("Valor de la métrica")
        st.pyplot(fig)

        st.markdown('<p class="metric-subtitle">Tiempo de ejecución ⏱️</p>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=df_modelos.sort_values("Tiempo (s)", ascending=True),
            y="Modelo",
            x="Tiempo (s)",
            ax=ax2,
            palette=["#ffb38a", "#ff8c42", "#ff6a21", "#ff4500"]
        )
        ax2.set_title("Tiempo de ejecución por modelo")
        ax2.set_xlabel("Tiempo (s)")
        ax2.set_ylabel("")
        st.pyplot(fig2)

    best_model = df_modelos.sort_values("F1", ascending=False).iloc[0]["Modelo"]
    st.success(f"Modelo seleccionado para despliegue: {best_model}")

elif pagina == "Predicción de Comentarios":
    st.title("📝 Predicción de Comentarios")
    st.write("Modelo cargado: **LinearSVC_TFIDF**")

    @st.cache_resource
    def load_model():
        try:
            return joblib.load("model_svc_tfidf.joblib")
        except:
            return None

    model = load_model()

    if model is None:
        st.error("No se pudo cargar el modelo. Verifique que `model_svc_tfidf.joblib` esté en el directorio.")
    else:
        st.subheader("Predicción individual 🗣️")
        user_input = st.text_area("Comentario a evaluar:", height=150)

        if st.button("Predecir comentario"):
            if user_input.strip() == "":
                st.warning("Ingrese un comentario.")
            else:
                pred = model.predict([user_input])[0]
                if pred == 1:
                    st.error("Clasificación: violación de reglas.")
                else:
                    st.success("Clasificación: comentario permitido.")

        st.markdown("---")
        st.subheader("Clasificación por archivo 📂")

        file = st.file_uploader("Sube un archivo CSV o TXT con comentarios", type=["csv", "txt"])

        if file is not None:
            file_type = file.name.split(".")[-1].lower()

            if file_type == "csv":
                df_in = pd.read_csv(file)
                col_texto = st.selectbox("Columna que contiene el texto:", df_in.columns)
                textos = df_in[col_texto].astype(str).tolist()
            else:  # txt
                stringio = StringIO(file.getvalue().decode("utf-8"))
                lines = [line.strip() for line in stringio.readlines() if line.strip()]
                df_in = pd.DataFrame({"texto": lines})
                col_texto = "texto"
                textos = df_in[col_texto].astype(str).tolist()

            if st.button("Clasificar archivo"):
                if len(textos) == 0:
                    st.warning("El archivo no contiene texto válido.")
                else:
                    preds = model.predict(textos)
                    df_result = df_in.copy()
                    df_result["prediccion"] = preds
                    df_result["etiqueta"] = df_result["prediccion"].map(
                        {0: "No viola reglas", 1: "Viola reglas"}
                    )

                    st.markdown("### Resultados de la clasificación")
                    st.dataframe(df_result, use_container_width=True)

                    out_csv = df_result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Descargar resultados en CSV",
                        data=out_csv,
                        file_name="resultados_clasificacion.csv",
                        mime="text/csv"
                    )

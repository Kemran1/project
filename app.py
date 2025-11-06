import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Анализ данных - Первичный анализ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 6px;
        padding: 1rem;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 6px;
        padding: 1rem;
        color: #856404;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 6px;
        padding: 1.5rem;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-header">Анализ данных</div>', unsafe_allow_html=True)
st.markdown("### Система первичного анализа и верификации данных")


def basic_data_info(df):

    st.markdown('<div class="section-header">Базовые характеристики данных</div>', unsafe_allow_html=True)


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Объем данных", f"{df.shape[0]:,}", "наблюдений")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Переменные", f"{df.shape[1]:,}", "столбцов")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Числовые данные", f"{numeric_cols:,}", "переменных")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        categorical_cols = df.select_dtypes(include=['object']).shape[1]
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Категориальные данные", f"{categorical_cols:,}", "переменных")
        st.markdown('</div>', unsafe_allow_html=True)


    st.subheader("Структура и метаданные")
    info_df = pd.DataFrame({
        'Переменная': df.columns,
        'Тип данных': df.dtypes,
        'Уникальные значения': df.nunique(),
        'Пропущенные значения': df.isnull().sum(),
        'Доля пропусков, %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(info_df, use_container_width=True, height=400)


def numeric_analysis(df):
    """Анализ числовых данных"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.markdown('<div class="warning-box">Числовые переменные для анализа отсутствуют</div>',
                    unsafe_allow_html=True)
        return

    st.markdown('<div class="section-header">Анализ числовых переменных</div>', unsafe_allow_html=True)


    st.subheader("Статистические показатели")
    desc_stats = df[numeric_cols].describe()

    desc_stats_ru = desc_stats.rename(index={
        'count': 'Количество',
        'mean': 'Среднее',
        'std': 'Стандартное отклонение',
        'min': 'Минимум',
        '25%': '25-й процентиль',
        '50%': 'Медиана',
        '75%': '75-й процентиль',
        'max': 'Максимум'
    })
    st.dataframe(desc_stats_ru, use_container_width=True)


    st.subheader("Визуальный анализ распределений")

    selected_cols = st.multiselect(
        "Выберите переменные для детального анализа:",
        options=numeric_cols.tolist(),
        default=numeric_cols[:min(2, len(numeric_cols))].tolist(),
        help="Выберите числовые переменные для построения графиков"
    )

    if selected_cols:

        if len(selected_cols) <= 4:
            st.write("**Распределение значений**")
            cols = st.columns(len(selected_cols))
            for idx, col in enumerate(selected_cols):
                with cols[idx]:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    df[col].hist(bins=20, ax=ax, alpha=0.7, color='#1f77b4')
                    ax.set_title(f'Распределение: {col}', fontsize=10)
                    ax.set_xlabel(col, fontsize=9)
                    ax.set_ylabel('Частота', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)


        st.write("**Анализ выбросов**")
        fig, ax = plt.subplots(figsize=(12, 6))
        df[selected_cols].boxplot(ax=ax)
        ax.set_title('Диаграммы размаха для анализа выбросов')
        ax.set_ylabel('Значения')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)


        if len(selected_cols) > 1:
            st.write("**Матрица корреляций**")
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = df[selected_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                        fmt='.2f', ax=ax, cbar_kws={'label': 'Коэффициент корреляции'})
            ax.set_title('Матрица корреляций между переменными')
            st.pyplot(fig)


def categorical_analysis(df):

    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) == 0:
        st.markdown('<div class="warning-box">⚠️ Категориальные переменные для анализа отсутствуют</div>',
                    unsafe_allow_html=True)
        return

    st.markdown('<div class="section-header">🏷️ Анализ категориальных переменных</div>', unsafe_allow_html=True)

    selected_cat_col = st.selectbox(
        "Выберите категориальную переменную для анализа:",
        options=categorical_cols.tolist(),
        help="Выберите переменную для анализа распределения категорий"
    )

    if selected_cat_col:
        value_counts = df[selected_cat_col].value_counts().head(15)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("**📊 Распределение категорий**")


            fig = px.bar(x=value_counts.index,
                         y=value_counts.values,
                         title=f'Распределение переменной: {selected_cat_col}',
                         labels={'x': 'Категории', 'y': 'Количество наблюдений'},
                         color=value_counts.values,
                         color_continuous_scale='Viridis')

            fig.update_layout(
                height=500,
                showlegend=False,
                xaxis_tickangle=-45,
                template='plotly_white'
            )


            fig.update_traces(texttemplate='%{y}', textposition='outside')

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("**📈 Статистика распределения**")
            freq_table = pd.DataFrame({
                'Категория': value_counts.index,
                'Абсолютная частота': value_counts.values,
                'Относительная частота, %': (value_counts.values / len(df) * 100).round(2)
            })

            styled_freq_table = freq_table.style.background_gradient(
                subset=['Абсолютная частота'],
                cmap='Blues'
            )

            st.dataframe(styled_freq_table, use_container_width=True, height=400)


            st.markdown("**📊 Общая статистика:**")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Всего категорий", f"{df[selected_cat_col].nunique()}")
            with col_b:
                st.metric("Наиболее частая", value_counts.index[0])

            st.metric("Доля наиболее частой",
                      f"{(value_counts.values[0] / len(df) * 100):.1f}%")


def missing_values_analysis(df):

    missing_total = df.isnull().sum().sum()
    if missing_total == 0:
        st.markdown('<div class="success-box">Пропущенные значения в данных отсутствуют</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="section-header">Анализ пропущенных значений</div>', unsafe_allow_html=True)

    missing_series = df.isnull().sum()
    missing_series = missing_series[missing_series > 0]

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Детализация пропусков по переменным**")
        missing_df = pd.DataFrame({
            'Переменная': missing_series.index,
            'Количество пропусков': missing_series.values,
            'Доля пропусков, %': (missing_series.values / len(df) * 100).round(2)
        }).sort_values('Количество пропусков', ascending=False)

        st.dataframe(missing_df, use_container_width=True)

    with col2:
        st.write("**Визуализация распределения пропусков**")
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_series.sort_values(ascending=False).plot(kind='bar', ax=ax, color='#ff6b6b')
        ax.set_title('Распределение пропущенных значений по переменным')
        ax.set_ylabel('Количество пропусков')
        ax.set_xlabel('Переменные')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)


def data_quality_checks(df):
    """Проверки качества данных"""
    st.markdown('<div class="section-header">Диагностика качества данных</div>', unsafe_allow_html=True)

    issues = []
    warnings_list = []
    info_list = []


    duplicates = df.duplicated().sum()
    if duplicates > 0:
        warnings_list.append(f"Обнаружено полных дубликатов записей: {duplicates}")


    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).sum() > 0:
            warnings_list.append(f"Обнаружены бесконечные значения в переменной: '{col}'")


    for col in df.columns:
        if df[col].nunique() == 1:
            info_list.append(f"Переменная '{col}' содержит постоянное значение")


    for col in df.columns:
        missing_percent = (df[col].isnull().sum() / len(df)) * 100
        if missing_percent > 50:
            warnings_list.append(f"Критический уровень пропусков в переменной '{col}': {missing_percent:.1f}%")

    if warnings_list:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.write("**Требуют внимания:**")
        for warning in warnings_list:
            st.write(f"• {warning}")
        st.markdown('</div>', unsafe_allow_html=True)

    if info_list:
        st.write("**Информационные сообщения:**")
        for info in info_list:
            st.write(f"• {info}")

    if not warnings_list and not info_list:
        st.markdown('<div class="success-box">Качество данных соответствует требованиям для анализа</div>',
                    unsafe_allow_html=True)


def export_analysis(df):

    st.markdown('<div class="section-header">Экспорт результатов анализа</div>', unsafe_allow_html=True)


    report = f"""
ОТЧЕТ ПЕРВИЧНОГО АНАЛИЗА ДАННЫХ
{'=' * 50}

ОБЩАЯ ИНФОРМАЦИЯ:
• Объем данных: {df.shape[0]:,} наблюдений
• Количество переменных: {df.shape[1]:,}
• Числовые переменные: {df.select_dtypes(include=[np.number]).shape[1]:,}
• Категориальные переменные: {df.select_dtypes(include=['object']).shape[1]:,}

КАЧЕСТВО ДАННЫХ:
• Всего пропущенных значений: {df.isnull().sum().sum():,}
• Полных дубликатов записей: {df.duplicated().sum():,}
• Переменные с пропусками: {', '.join(df.columns[df.isnull().sum() > 0].tolist()) if df.isnull().sum().sum() > 0 else 'отсутствуют'}

СТАТИСТИЧЕСКИЕ ХАРАКТЕРИСТИКИ:
{df.describe().to_string()}

СГЕНЕРИРОВАНО: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    st.text_area("Полный отчет анализа", report, height=400)

    st.write("**Загрузка результатов:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Статистический отчет", use_container_width=True):
            desc_stats = df.describe()
            csv = desc_stats.to_csv()
            st.download_button(
                label="Скачать CSV",
                data=csv,
                file_name="statistical_report.csv",
                mime="text/csv",
                use_container_width=True
            )

    with col2:
        if st.button("Метаданные", use_container_width=True):
            info_data = pd.DataFrame({
                'Переменная': df.columns,
                'Тип данных': df.dtypes,
                'Уникальные значения': df.nunique(),
                'Пропуски': df.isnull().sum()
            })
            csv = info_data.to_csv(index=False)
            st.download_button(
                label="Скачать CSV",
                data=csv,
                file_name="metadata.csv",
                mime="text/csv",
                use_container_width=True
            )

    with col3:
        if st.button("Полный отчет", use_container_width=True):
            st.download_button(
                label="Скачать TXT",
                data=report,
                file_name="data_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )



st.markdown("---")
uploaded_file = st.file_uploader(
    "Загрузите файл с данными для анализа",
    type=['csv', 'xlsx'],
    help="Поддерживаются файлы формата CSV и Excel (XLSX)"
)

if uploaded_file is not None:
    try:

        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.markdown('<div class="section-header">Предварительный просмотр данных</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

        basic_data_info(df)
        numeric_analysis(df)
        categorical_analysis(df)
        missing_values_analysis(df)
        data_quality_checks(df)
        export_analysis(df)

    except Exception as e:
        st.error(f"Ошибка при обработке файла: {str(e)}")
        st.info("Убедитесь, что файл имеет корректный формат и кодировку")

else:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write("### Начало работы")
    st.write("""
    Для запуска анализа загрузите файл с данными в формате CSV или Excel.

    **Возможности системы:**
    • Автоматический анализ структуры данных
    • Статистическая характеристика числовых переменных
    • Анализ распределения категориальных данных
    • Диагностика качества данных и пропущенных значений
    • Визуализация основных закономерностей
    • Экспорт результатов анализа в различные форматы
    """)
    st.markdown('</div>', unsafe_allow_html=True)
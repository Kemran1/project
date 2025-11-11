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
import time

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

    /* Стили для анимированного заголовка */
    .typewriter-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    .typewriter h1 {
        font-size: 2.8rem;
        color: #1f77b4;
        font-weight: 400;
        overflow: hidden;
        border-right: .15em solid #1f77b4;
        white-space: nowrap;
        margin: 0 auto;
        letter-spacing: .10em;
        animation: 
            typing 3.5s steps(40, end),
            blink-caret .75s step-end infinite;
    }

    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }

    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: #1f77b4; }
    }

    /* Стили для бегущей строки с изменяющимся текстом */
    .rotating-text {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        font-weight: 400;
        height: 80px;
        margin-bottom: 2rem;
    }

    .text-item {
        position: absolute;
        width: 100%;
        text-align: center;
        opacity: 0;
        animation: rotateWord 18s linear infinite 0s;
    }

    .text-item:nth-child(2) { animation-delay: 3s; }
    .text-item:nth-child(3) { animation-delay: 6s; }
    .text-item:nth-child(4) { animation-delay: 9s; }
    .text-item:nth-child(5) { animation-delay: 12s; }
    .text-item:nth-child(6) { animation-delay: 15s; }

    @keyframes rotateWord {
        0% { opacity: 0; transform: translateY(20px); }
        2% { opacity: 1; transform: translateY(0px); }
        15% { opacity: 1; transform: translateY(0px); }
        18% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)



def create_animated_header():
    st.markdown("""
    <div class="rotating-text">
        <div class="text-item">📊 Анализ данных</div>
        <div class="text-item">🔍 Исследование данных</div>
        <div class="text-item">📈 Визуализация</div>
        <div class="text-item">📋 Отчетность</div>
        <div class="text-item">🎯 Инсайты</div>
        <div class="text-item">🚀 Аналитика</div>
    </div>
    """, unsafe_allow_html=True)



create_animated_header()
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




def enhanced_numeric_analysis(df):

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.markdown('<div class="warning-box">Числовые переменные для анализа отсутствуют</div>',
                    unsafe_allow_html=True)
        return

    st.markdown('<div class="section-header">📊 Расширенный анализ числовых переменных</div>', unsafe_allow_html=True)


    selected_cols = st.multiselect(
        "Выберите переменные для детального анализа:",
        options=numeric_cols.tolist(),
        default=numeric_cols[:min(3, len(numeric_cols))].tolist(),
        help="Выберите числовые переменные для построения графиков"
    )

    if not selected_cols:
        return


    tab1, tab2, tab3, tab4 = st.tabs(["📈 Распределения", "📊 Сравнение", "🔄 Корреляции", "📋 Статистика"])

    with tab1:

        st.subheader("Распределения с плотностью вероятности")
        cols = st.columns(2)
        for i, col in enumerate(selected_cols):
            with cols[i % 2]:

                fig = px.histogram(df, x=col, nbins=30,
                                   title=f'📊 {col} - Распределение',
                                   color_discrete_sequence=['#1f77b4'],
                                   opacity=0.7,
                                   marginal="box")

                fig.update_layout(
                    height=400,
                    showlegend=False,
                    template='plotly_white',
                    font=dict(size=12),
                    title_font=dict(size=14)
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:

        st.subheader("Анализ выбросов и распределений")
        col1, col2 = st.columns(2)

        with col1:

            fig_box = px.box(df, y=selected_cols,
                             title='📦 Диаграммы размаха',
                             color_discrete_sequence=['#ff7f0e'])
            fig_box.update_layout(height=500, template='plotly_white')
            st.plotly_chart(fig_box, use_container_width=True)

        with col2:

            if len(selected_cols) <= 4:
                fig_violin = px.violin(df, y=selected_cols,
                                       title='🎻 Violin plot (плотность распределения)',
                                       box=True,
                                       color_discrete_sequence=['#2ca02c'])
                fig_violin.update_layout(height=500, template='plotly_white')
                st.plotly_chart(fig_violin, use_container_width=True)

    with tab3:

        st.subheader("Анализ взаимосвязей")

        if len(selected_cols) > 1:
            col1, col2 = st.columns([2, 1])

            with col1:

                corr_matrix = df[selected_cols].corr()
                fig_heatmap = px.imshow(corr_matrix,
                                        title='🔥 Тепловая карта корреляций',
                                        color_continuous_scale='RdBu_r',
                                        aspect='auto',
                                        text_auto=True)
                fig_heatmap.update_layout(height=500)
                st.plotly_chart(fig_heatmap, use_container_width=True)

            with col2:

                if len(selected_cols) <= 5:
                    fig_scatter_matrix = px.scatter_matrix(df[selected_cols],
                                                           title='🔄 Матрица scatter plot',
                                                           height=600)
                    st.plotly_chart(fig_scatter_matrix, use_container_width=True)


        if len(selected_cols) >= 2:
            st.subheader("Диаграмма рассеяния")
            scatter_col1, scatter_col2, scatter_col3 = st.columns(3)

            with scatter_col1:
                x_axis = st.selectbox("Ось X:", selected_cols, index=0)
            with scatter_col2:
                y_axis = st.selectbox("Ось Y:", selected_cols, index=1)
            with scatter_col3:
                color_by = st.selectbox("Цвет по:", ["Нет"] + selected_cols)

            fig_scatter = px.scatter(df, x=x_axis, y=y_axis,
                                     color=color_by if color_by != "Нет" else None,
                                     title=f'📊 {x_axis} vs {y_axis}',
                                     trendline="lowess",
                                     opacity=0.6)
            fig_scatter.update_layout(height=500, template='plotly_white')
            st.plotly_chart(fig_scatter, use_container_width=True)

    with tab4:

        st.subheader("Детальная статистика")


        desc_stats = df[selected_cols].describe().T
        desc_stats['skewness'] = df[selected_cols].skew()
        desc_stats['kurtosis'] = df[selected_cols].kurtosis()
        desc_stats_ru = desc_stats.rename(columns={
            'count': 'Количество',
            'mean': 'Среднее',
            'std': 'Стандартное отклонение',
            'min': 'Минимум',
            '25%': '25-й процентиль',
            '50%': 'Медиана',
            '75%': '75-й процентиль',
            'max': 'Максимум',
            'skewness': 'Асимметрия',
            'kurtosis': 'Эксцесс'
        })

        st.dataframe(desc_stats_ru.style.background_gradient(cmap='Blues'),
                     use_container_width=True)


def enhanced_categorical_analysis(df):

    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) == 0:
        st.markdown('<div class="warning-box">⚠️ Категориальные переменные для анализа отсутствуют</div>',
                    unsafe_allow_html=True)
        return

    st.markdown('<div class="section-header">🏷️ Расширенный анализ категориальных переменных</div>',
                unsafe_allow_html=True)

    selected_cat_col = st.selectbox(
        "Выберите категориальную переменную для анализа:",
        options=categorical_cols.tolist(),
        help="Выберите переменную для анализа распределения категорий"
    )

    if not selected_cat_col:
        return

    value_counts = df[selected_cat_col].value_counts().head(15)


    tab1, tab2, tab3 = st.tabs(["📊 Основные графики", "🎯 Детальный анализ", "📈 Сравнение"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:

            fig_pie = px.pie(values=value_counts.values,
                             names=value_counts.index,
                             title=f'🎯 Распределение: {selected_cat_col}',
                             color_discrete_sequence=px.colors.sequential.Viridis)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:

            fig_bar_h = px.bar(x=value_counts.values,
                               y=value_counts.index,
                               orientation='h',
                               title='📊 Топ категорий',
                               color=value_counts.values,
                               color_continuous_scale='Viridis')
            fig_bar_h.update_layout(height=500, showlegend=False,
                                    xaxis_title="Количество",
                                    yaxis_title="Категории")
            st.plotly_chart(fig_bar_h, use_container_width=True)

    with tab2:

        if len(value_counts) > 5:
            fig_treemap = px.treemap(names=value_counts.index,
                                     parents=[''] * len(value_counts),
                                     values=value_counts.values,
                                     title='🗺️ Treemap распределения')
            fig_treemap.update_layout(height=500)
            st.plotly_chart(fig_treemap, use_container_width=True)


        if len(value_counts) >= 8:
            fig_sunburst = px.sunburst(names=value_counts.index,
                                       parents=[''] * len(value_counts),
                                       values=value_counts.values,
                                       title='☀️ Sunburst диаграмма')
            fig_sunburst.update_layout(height=500)
            st.plotly_chart(fig_sunburst, use_container_width=True)

    with tab3:

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            compare_with = st.selectbox("Сравнить с числовой переменной:",
                                        numeric_cols.tolist())

            if compare_with:
                col1, col2 = st.columns(2)

                with col1:

                    fig_box_cat = px.box(df, x=selected_cat_col, y=compare_with,
                                         title=f'📦 {compare_with} по категориям')
                    fig_box_cat.update_layout(height=500,
                                              xaxis_tickangle=-45,
                                              template='plotly_white')
                    st.plotly_chart(fig_box_cat, use_container_width=True)

                with col2:

                    avg_by_cat = df.groupby(selected_cat_col)[compare_with].mean().sort_values(ascending=False).head(10)
                    fig_bar_avg = px.bar(x=avg_by_cat.index, y=avg_by_cat.values,
                                         title=f'📊 Среднее {compare_with} по категориям',
                                         color=avg_by_cat.values,
                                         color_continuous_scale='Viridis')
                    fig_bar_avg.update_layout(height=500,
                                              xaxis_tickangle=-45,
                                              showlegend=False)
                    st.plotly_chart(fig_bar_avg, use_container_width=True)


def create_advanced_dashboard(df):

    st.markdown('<div class="section-header">🚀 Продвинутая аналитика</div>', unsafe_allow_html=True)


    st.subheader("📋 Быстрый обзор данных")


    analysis_type = st.radio(
        "Тип анализа:",
        ["📊 Общий обзор", "📈 Тренды", "🔍 Аномалии", "📋 Сводка"],
        horizontal=True
    )

    if analysis_type == "📊 Общий обзор":
        create_overview_dashboard(df)
    elif analysis_type == "📈 Тренды":
        create_trends_dashboard(df)
    elif analysis_type == "🔍 Аномалии":
        create_anomalies_dashboard(df)
    else:
        create_summary_dashboard(df)


def create_overview_dashboard(df):

    col1, col2 = st.columns(2)

    with col1:

        type_counts = df.dtypes.value_counts()
        fig_types = px.pie(values=type_counts.values,
                           names=type_counts.index.astype(str),
                           title='📊 Распределение типов данных',
                           color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_types, use_container_width=True)

    with col2:

        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            missing_data = missing_data[missing_data > 0]
            fig_missing = px.bar(x=missing_data.index, y=missing_data.values,
                                 title='⚠️ Распределение пропусков',
                                 color=missing_data.values,
                                 color_continuous_scale='Reds')
            fig_missing.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ Пропущенные значения отсутствуют")


def create_trends_dashboard(df):
    """Панель анализа трендов"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            x_col = st.selectbox("Ось X для тренда:", numeric_cols)
        with col2:
            y_col = st.selectbox("Ось Y для тренда:", numeric_cols)


        fig_trend = px.scatter(df, x=x_col, y=y_col,
                               trendline="ols",
                               title=f'📈 Тренд: {x_col} vs {y_col}',
                               opacity=0.6)
        fig_trend.update_layout(height=500)
        st.plotly_chart(fig_trend, use_container_width=True)


def create_anomalies_dashboard(df):

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Выберите переменную для анализа аномалий:", numeric_cols)


        Q1 = df[selected_col].quantile(0.25)
        Q3 = df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        anomalies = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Всего наблюдений", len(df))
            st.metric("Аномалий обнаружено", len(anomalies))

        with col2:
            st.metric("Доля аномалий", f"{(len(anomalies) / len(df) * 100):.2f}%")
            st.metric("Границы", f"[{lower_bound:.2f}, {upper_bound:.2f}]")


        fig_anomalies = px.scatter(df, x=df.index, y=selected_col,
                                   title=f'🔍 Обнаружение аномалий в {selected_col}',
                                   color=((df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)),
                                   color_discrete_map={True: 'red', False: 'blue'})
        fig_anomalies.update_layout(height=500)
        st.plotly_chart(fig_anomalies, use_container_width=True)


def create_summary_dashboard(df):

    st.subheader("📋 Ключевые показатели")


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Наблюдения", f"{len(df):,}")

    with col2:
        st.metric("🔢 Переменные", f"{len(df.columns):,}")

    with col3:
        numeric_count = df.select_dtypes(include=[np.number]).shape[1]
        st.metric("📈 Числовые", f"{numeric_count}")

    with col4:
        categorical_count = df.select_dtypes(include=['object']).shape[1]
        st.metric("🏷️ Категориальные", f"{categorical_count}")


    st.subheader("⚡ Быстрая статистика")
    quick_stats = df.describe().T[['mean', 'std', 'min', 'max']].round(2)
    st.dataframe(quick_stats.style.background_gradient(cmap='YlOrBr'),
                 use_container_width=True)

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
    type=['csv', 'xlsx', 'xls'],
    help="Поддерживаются файлы формата CSV, Excel (XLSX) и старые файлы Excel (XLS)"
)


if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file, engine='xlrd')
        else:
            st.error("Неподдерживаемый формат файла")
            st.stop()

        st.markdown('<div class="section-header">Предварительный просмотр данных</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)


        basic_data_info(df)
        enhanced_numeric_analysis(df)
        enhanced_categorical_analysis(df)
        create_advanced_dashboard(df)
        missing_values_analysis(df)
        data_quality_checks(df)
        export_analysis(df)

    except Exception as e:
        st.error(f"Ошибка при обработке файла: {str(e)}")
        st.info("Убедитесь, что файл имеет корректный формат и кодировку")

else:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write("### Начало работы")
    st.write("Для запуска анализа загрузите файл с данными в формате CSV или Excel.")

    st.write("**Основные возможности системы:**")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Анализ структуры данных**")
        st.write("Автоматическое определение типов данных, анализ размерности и метаданных наборов данных")

        st.write("**Статистический анализ**")
        st.write("Расчет описательной статистики для числовых показателей, анализ распределений и выбросов")

        st.write("**Категориальный анализ**")
        st.write("Анализ частотности категорий, распределение текстовых данных")

    with col2:
        st.write("**Диагностика качества**")
        st.write("Выявление пропущенных значений, дубликатов и аномалий в данных")

        st.write("**Визуализация данных**")
        st.write("Построение графиков, диаграмм и тепловых карт для наглядного представления")

        st.write("**Экспорт результатов**")
        st.write("Сохранение отчетов в различных форматах для дальнейшего использования")

    st.write("**Примеры визуализаций, доступных в системе:**")

    fig_col1, fig_col2, fig_col3 = st.columns(3)

    with fig_col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        np.random.seed(42)
        sample_data = np.random.normal(100, 15, 1000)
        ax1.hist(sample_data, bins=20, alpha=0.7, color='#1f77b4')
        ax1.set_title('Гистограмма распределения')
        ax1.set_xlabel('Значения')
        ax1.set_ylabel('Частота')
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    with fig_col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sample_data = [np.random.normal(100, 15, 100) for _ in range(3)]
        ax2.boxplot(sample_data)
        ax2.set_title('Диаграмма размаха')
        ax2.set_ylabel('Значения')
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    with fig_col3:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        categories = ['A', 'B', 'C', 'D', 'E']
        values = [25, 40, 30, 35, 20]
        ax3.bar(categories, values, color='#2e86ab', alpha=0.7)
        ax3.set_title('Столбчатая диаграмма')
        ax3.set_xlabel('Категории')
        ax3.set_ylabel('Значения')
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

    st.write("**Поддерживаемые форматы данных:**")
    format_col1, format_col2, format_col3 = st.columns(3)

    with format_col1:
        st.write("• CSV файлы")
        st.write("• Excel (XLSX)")

    with format_col2:
        st.write("• Excel (XLS)")
        st.write("• Excel (XLSX)")

    with format_col3:
        st.write("• Статистические данные")
        st.write("• Другие табличные данные")

    st.write("")
    st.write("**Для начала работы:**")
    st.write("1. Подготовьте файл с данными в одном из поддерживаемых форматов")
    st.write("2. Загрузите файл с помощью формы выше")
    st.write("3. Дождитесь автоматического анализа данных")
    st.write("4. Изучите результаты в соответствующих разделах")

    st.markdown('</div>', unsafe_allow_html=True)

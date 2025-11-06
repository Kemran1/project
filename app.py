import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Настройки отображения
st.set_page_config(page_title="Помощник аналитика", layout="wide")
st.title("📊 Помощник для первичного анализа данных")


def basic_data_info(df):
    """Базовая информация о данных"""
    st.header("1. Базовая информация о данных")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Строки", df.shape[0])
    with col2:
        st.metric("Столбцы", df.shape[1])
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
        st.metric("Числовые", numeric_cols)
    with col4:
        categorical_cols = df.select_dtypes(include=['object']).shape[1]
        st.metric("Текстовые", categorical_cols)

    # Информация о типах данных
    st.subheader("Типы данных и пропуски")
    info_df = pd.DataFrame({
        'Столбец': df.columns,
        'Тип': df.dtypes,
        'Уникальных': df.nunique(),
        'Пропуски': df.isnull().sum(),
        '% пропусков': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(info_df, use_container_width=True)


def numeric_analysis(df):
    """Анализ числовых данных"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.warning("Нет числовых столбцов для анализа")
        return

    st.header("2. Анализ числовых данных")

    # Описательная статистика
    st.subheader("Описательная статистика")
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)

    # Визуализации
    st.subheader("Визуализации распределений")

    # Выбор столбцов для анализа
    selected_cols = st.multiselect(
        "Выберите столбцы для анализа:",
        options=numeric_cols.tolist(),
        default=numeric_cols[:2].tolist() if len(numeric_cols) >= 2 else numeric_cols.tolist()
    )

    if selected_cols:
        # Гистограммы
        if len(selected_cols) <= 4:
            cols = st.columns(len(selected_cols))
            for idx, col in enumerate(selected_cols):
                with cols[idx]:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    df[col].hist(bins=20, ax=ax, alpha=0.7)
                    ax.set_title(f'Распределение {col}')
                    ax.set_xlabel(col)
                    st.pyplot(fig)

        # Box plots
        st.subheader("Анализ выбросов (Box plots)")
        fig, ax = plt.subplots(figsize=(12, 6))
        df[selected_cols].boxplot(ax=ax)
        ax.set_title('Распределение и выбросы')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

        # Корреляционная матрица
        if len(selected_cols) > 1:
            st.subheader("Корреляционная матрица")
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = df[selected_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                        fmt='.2f', ax=ax)
            ax.set_title('Корреляции между переменными')
            st.pyplot(fig)


def categorical_analysis(df):
    """Анализ категориальных данных"""
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) == 0:
        st.warning("Нет категориальных столбцов для анализа")
        return

    st.header("3. Анализ категориальных данных")

    selected_cat_col = st.selectbox(
        "Выберите категориальный столбец для анализа:",
        options=categorical_cols.tolist()
    )

    if selected_cat_col:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Столбчатая диаграмма
            fig, ax = plt.subplots(figsize=(10, 6))
            value_counts = df[selected_cat_col].value_counts().head(15)
            value_counts.plot(kind='bar', ax=ax)
            ax.set_title(f'Распределение: {selected_cat_col}')
            ax.set_ylabel('Количество')
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with col2:
            # Таблица с частотами
            st.subheader("Топ значений")
            freq_table = pd.DataFrame({
                'Значение': value_counts.index,
                'Количество': value_counts.values,
                'Доля %': (value_counts.values / len(df) * 100).round(2)
            })
            st.dataframe(freq_table, use_container_width=True)


def missing_values_analysis(df):
    """Анализ пропущенных значений"""
    missing_total = df.isnull().sum().sum()
    if missing_total == 0:
        st.success("✅ Пропущенных значений не обнаружено")
        return

    st.header("4. Анализ пропущенных значений")

    # Общая информация
    missing_series = df.isnull().sum()
    missing_series = missing_series[missing_series > 0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Пропуски по столбцам")
        missing_df = pd.DataFrame({
            'Столбец': missing_series.index,
            'Пропуски': missing_series.values,
            'Доля %': (missing_series.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True)

    with col2:
        st.subheader("Визуализация пропусков")
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_series.plot(kind='bar', ax=ax)
        ax.set_title('Распределение пропущенных значений')
        ax.set_ylabel('Количество пропусков')
        plt.xticks(rotation=45)
        st.pyplot(fig)


def data_quality_checks(df):
    """Проверки качества данных"""
    st.header("5. Проверки качества данных")

    issues = []

    # Проверка на полные дубликаты
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"⚠️ Найдено {duplicates} полных дубликатов строк")

    # Проверка числовых столбцов на бесконечные значения
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).sum() > 0:
            issues.append(f"⚠️ В столбце '{col}' найдены бесконечные значения")

    # Проверка на постоянные столбцы
    for col in df.columns:
        if df[col].nunique() == 1:
            issues.append(f"ℹ️ Столбец '{col}' содержит только одно уникальное значение")

    # Проверка на столбцы с большим количеством пропусков
    for col in df.columns:
        missing_percent = (df[col].isnull().sum() / len(df)) * 100
        if missing_percent > 50:
            issues.append(f"⚠️ В столбце '{col}' более 50% пропусков ({missing_percent:.1f}%)")

    if issues:
        for issue in issues:
            st.write(issue)
    else:
        st.success("✅ Серьезных проблем с качеством данных не обнаружено")


def export_analysis(df):
    """Экспорт результатов анализа"""
    st.header("6. Экспорт результатов")

    # Создание отчета
    report = f"""
    ОТЧЕТ ПЕРВИЧНОГО АНАЛИЗА ДАННЫХ
    ================================

    Общая информация:
    - Количество наблюдений: {df.shape[0]}
    - Количество переменных: {df.shape[1]}
    - Числовых переменных: {df.select_dtypes(include=[np.number]).shape[1]}
    - Категориальных переменных: {df.select_dtypes(include=['object']).shape[1]}

    Пропущенные значения:
    - Всего пропусков: {df.isnull().sum().sum()}
    - Столбцы с пропусками: {df.columns[df.isnull().sum() > 0].tolist()}

    Дубликаты:
    - Полных дубликатов: {df.duplicated().sum()}
    """

    st.text_area("Текстовый отчет", report, height=300)

    # Кнопки для экспорта
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Скачать описательную статистику"):
            desc_stats = df.describe()
            csv = desc_stats.to_csv()
            st.download_button(
                label="Скачать CSV",
                data=csv,
                file_name="descriptive_statistics.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("📊 Скачать информацию о данных"):
            info_data = pd.DataFrame({
                'Столбец': df.columns,
                'Тип': df.dtypes,
                'Уникальных': df.nunique(),
                'Пропуски': df.isnull().sum()
            })
            csv = info_data.to_csv(index=False)
            st.download_button(
                label="Скачать CSV",
                data=csv,
                file_name="data_info.csv",
                mime="text/csv"
            )


# Основной интерфейс
uploaded_file = st.file_uploader(
    "📁 Загрузите файл с данными (CSV или Excel)",
    type=['csv', 'xlsx']
)

if uploaded_file is not None:
    try:
        # Загрузка данных
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Показ первых строк данных
        st.header("Предпросмотр данных")
        st.dataframe(df.head(), use_container_width=True)

        # Выполнение анализа
        basic_data_info(df)
        numeric_analysis(df)
        categorical_analysis(df)
        missing_values_analysis(df)
        data_quality_checks(df)
        export_analysis(df)

    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {str(e)}")

else:
    st.info("""
    👆 **Загрузите файл для начала анализа**

    Этот инструмент поможет с:
    • Быстрым ознакомлением с данными
    • Анализом распределений и выбросов
    • Поиском пропущенных значений
    • Проверкой качества данных
    • Созданием базовых визуализаций
    """)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, t
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


plt.style.use('ggplot')
sns.set_palette("Set2")

print("\n1. ЗАГРУЗКА И ОЧИСТКА ДАННЫХ")
df = pd.read_excel('Квартиры.xlsx', sheet_name='ЦИАН - Продажа городской')
df.columns = df.columns.str.strip()
df['price'] = df['price'].astype(str).str.replace(' ', '').str.replace('\xa0', '').astype(float)
df = df.dropna(subset=['price', 'total_area', 'rooms', 'floor', 'total_floors', 'metro_min', 'ceiling_height'])
df = df[df['ceiling_height'] < 10]
df['first_floor'] = (df['floor'] == 1).astype(int)

print("\n2. ОПИСАТЕЛЬНАЯ СТАТИСТИКА")
numeric_cols = ['price', 'rooms', 'metro_min', 'total_area', 'floor', 'total_floors', 'ceiling_height']
desc = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).round(2)
desc.loc['mode'] = df[numeric_cols].mode().iloc[0]
print(desc)

print("\n3. ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЙ (сохранение в файлы)")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].hist(df['price'] / 1_000_000, bins=20, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Гистограмма цен')
axes[0, 0].set_xlabel('Цена, млн руб.')

axes[0, 1].hist(df['total_area'], bins=20, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Гистограмма общей площади')
axes[0, 1].set_xlabel('Площадь, м²')

sns.boxplot(y=df['price'] / 1_000_000, ax=axes[1, 0], color='skyblue')
axes[1, 0].set_title('Размах цен')
axes[1, 0].set_ylabel('Цена, млн руб.')

sns.boxplot(y=df['total_area'], ax=axes[1, 1], color='lightgreen')
axes[1, 1].set_title('Размах общей площади')
axes[1, 1].set_ylabel('Площадь, м²')

plt.tight_layout()
plt.savefig('descriptive_plots.png', dpi=300)
plt.show()


print("\n4. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
corr_cols = ['price', 'rooms', 'metro_min', 'total_area', 'floor', 'total_floors', 'ceiling_height']
corr_matrix = df[corr_cols].corr()

plt.rcParams['font.family'] = 'Arial'

ru_labels_short = {
    'price': 'Цена',
    'rooms': 'Комнаты',
    'metro_min': 'Метро (мин)',
    'total_area': 'Площадь',
    'floor': 'Этаж',
    'total_floors': 'Этажность дома',
    'ceiling_height': 'Потолки'
}
corr_matrix_ru = corr_matrix.rename(index=ru_labels_short, columns=ru_labels_short)

plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix_ru, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, annot_kws={"size": 11})
plt.title('Матрица корреляций Пирсона', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()

print("\nПроверка значимости корреляций с ценой (t-критерий):")
alpha = 0.05
t_crit = t.ppf(1 - alpha/2, df=len(df)-2)
for col in corr_cols[1:]:
    r, p = pearsonr(df['price'], df[col])
    t_stat = r * np.sqrt((len(df) - 2) / (1 - r**2))
    significant = 'ДА' if p < alpha else 'НЕТ'
    print(f"{col:15} r = {r:.3f}, t = {t_stat:.2f}, p = {p:.4f} -> значим: {significant}")

pairs_ru = [
    ('total_area', 'Общая площадь, м²'),
    ('rooms', 'Количество комнат'),
    ('total_floors', 'Этажность дома (всего этажей)'),
    ('ceiling_height', 'Высота потолков, м'),
    ('floor', 'Этаж расположения квартиры'),
    ('metro_min', 'Время до метро, мин')
]

fig, axes = plt.subplots(3, 2, figsize=(14, 18))

for ax, (x_col, x_label) in zip(axes.flatten(), pairs_ru):
    sns.scatterplot(x=df[x_col], y=df['price'] / 1_000_000, ax=ax, alpha=0.5, color='teal', s=40)
    ax.set_title(f'Связь цены и параметра: {x_label.split(",")[0]}', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('Цена квартиры, млн руб.', fontsize=11)
   
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('scatter_plots_all.png', dpi=300) 
plt.show()
print("\n5. РЕГРЕССИОННЫЙ АНАЛИЗ (5 моделей)")

plt.rcParams['font.family'] = 'Arial'
X1 = ['total_area']
X2 = ['total_area', 'rooms', 'metro_min']
X3 = ['total_area', 'metro_min', 'first_floor']
X4 = ['total_area', 'ceiling_height', 'rooms', 'total_floors']
X5 = ['total_area', 'metro_min', 'first_floor', 'total_floors']  

model_formulas_ru = {
    "Модель 1": r"Цена = $\beta_0$ + $\beta_1 \cdot$ Площадь",
    "Модель 2": r"Цена = $\beta_0$ + $\beta_1 \cdot$ Площадь + $\beta_2 \cdot$ Комнаты + $\beta_3 \cdot$ Метро",
    "Модель 3": r"Цена = $\beta_0$ + $\beta_1 \cdot$ Площадь + $\beta_2 \cdot$ Метро + $\beta_3 \cdot$ Первый_этаж",
    "Модель 4": r"Цена = $\beta_0$ + $\beta_1 \cdot$ Площадь + $\beta_2 \cdot$ Потолки + $\beta_3 \cdot$ Комнаты + $\beta_4 \cdot$ Этажность",
    "Модель 5": r"ln(Цена) = $\beta_0$ + $\beta_1 \cdot$ ln(Площадь) + $\beta_2 \cdot$ Метро + $\beta_3 \cdot$ Первый_этаж + $\beta_4 \cdot$ Этажность"
}

def run_regression(X_cols, target_log=False, model_name='Model'):
    X = df[X_cols].copy()
    if target_log:
        X = X.assign(log_total_area=np.log(df['total_area']))
        X = X.drop('total_area', axis=1)  
        y = np.log(df['price'])
        X_cols_display = ['log_total_area'] + [c for c in X_cols if c != 'total_area']
    else:
        y = df['price']
        X_cols_display = X_cols
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit()
    print(f"\n{model_name}")
    print(f"R² = {model.rsquared:.4f}, Adj.R² = {model.rsquared_adj:.4f}")
    print(model.summary())
    return model, X_cols_display

m1, cols1 = run_regression(X1, target_log=False, model_name="Модель 1")
m2, cols2 = run_regression(X2, target_log=False, model_name="Модель 2")
m3, cols3 = run_regression(X3, target_log=False, model_name="Модель 3")
m4, cols4 = run_regression(X4, target_log=False, model_name="Модель 4")
m5, cols5 = run_regression(X5, target_log=True, model_name="Модель 5")

models = [m1, m2, m3, m4, m5]
names = ['Модель 1', 'Модель 2', 'Модель 3', 'Модель 4', 'Модель 5']
adj_r2 = [m.rsquared_adj for m in models]

print("\n6. СРАВНЕНИЕ МОДЕЛЕЙ ПО СКОРРЕКТИРОВАННОМУ R²")
comparison = pd.DataFrame({'Модель': names, 'Adj. R²': adj_r2})
print(comparison)

best_idx = np.argmax(adj_r2)
print(f"\nЛучшая модель: {names[best_idx]} с Adj.R² = {adj_r2[best_idx]:.4f}")

print("\n7. ПОСТРОЕНИЕ ДИАГНОСТИЧЕСКИХ ГРАФИКОВ ДЛЯ ВСЕХ МОДЕЛЕЙ")

fig, axes = plt.subplots(5, 2, figsize=(16, 25))

for i, (model, name) in enumerate(zip(models, names)):

    y_pred = model.predict()
    y_actual = model.model.endog  

    if name == 'Модель 5':
        y_pred_plot = np.exp(y_pred)
        y_actual_plot = np.exp(y_actual)
        residuals = y_actual - y_pred 
    else:
        y_pred_plot = y_pred
        y_actual_plot = y_actual
        residuals = y_actual - y_pred

    ax_scatter = axes[i, 0]
    if name == 'Модель 5':
        ax_scatter.scatter(y_actual_plot, y_pred_plot, alpha=0.5, color='steelblue', edgecolor='k', s=25)
        ax_scatter.plot([y_actual_plot.min(), y_actual_plot.max()], [y_actual_plot.min(), y_actual_plot.max()], 'r--', lw=2)
    else:
        ax_scatter.scatter(y_actual_plot / 1_000_000, y_pred_plot / 1_000_000, alpha=0.5, color='steelblue', edgecolor='k', s=25)
        ax_scatter.plot([y_actual_plot.min() / 1_000_000, y_actual_plot.max() / 1_000_000], 
                        [y_actual_plot.min() / 1_000_000, y_actual_plot.max() / 1_000_000], 'r--', lw=2)
        
    ax_scatter.set_title(f'{name}\n{model_formulas_ru[name]}', fontsize=11, fontweight='bold')
    ax_scatter.set_xlabel('Фактические значения', fontsize=9)
    ax_scatter.set_ylabel('Прогнозные значения', fontsize=9)
    ax_scatter.grid(True, linestyle='--', alpha=0.5)
    ax_hist = axes[i, 1]
    ax_hist.hist(residuals, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax_hist.axvline(0, color='red', linestyle='--', lw=2)
    ax_hist.set_title(f'{name} — остатки', fontsize=11, fontweight='bold')
    ax_hist.set_xlabel('Остаток', fontsize=9)
    ax_hist.set_ylabel('Частота', fontsize=9)
    ax_hist.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('all_models_diagnostic_plots.png', dpi=300)
plt.show()

print("\nАнализ полностью завершён. Итоговый диагностический лист со всеми моделями сохранен как 'all_models_diagnostic_plots.png'.")
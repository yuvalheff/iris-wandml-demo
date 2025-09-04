import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# App color palette for consistent styling
app_color_palette = [
    'rgba(99, 110, 250, 0.8)',   # Blue
    'rgba(239, 85, 59, 0.8)',    # Red/Orange-Red
    'rgba(0, 204, 150, 0.8)',    # Green
    'rgba(171, 99, 250, 0.8)',   # Purple
    'rgba(255, 161, 90, 0.8)',   # Orange
    'rgba(25, 211, 243, 0.8)',   # Cyan
    'rgba(255, 102, 146, 0.8)',  # Pink
    'rgba(182, 232, 128, 0.8)',  # Light Green
    'rgba(255, 151, 255, 0.8)',  # Magenta
    'rgba(254, 203, 82, 0.8)'    # Yellow
]

# Load the training dataset
train_data = pd.read_csv('/Users/yuvalheffetz/ds-agent-projects/session_3e0a615d-9170-492c-b8ab-7121cbd38bd2/data/train_set.csv')

print("Dataset Shape:", train_data.shape)
print("\nDataset Info:")
train_data.info()
print("\nFirst 5 rows:")
print(train_data.head())

# Basic statistics and data types analysis
print("Data Types:")
print(train_data.dtypes)
print("\nMissing Values:")
print(train_data.isnull().sum())
print("\nBasic Statistics:")
print(train_data.describe())

# Target variable distribution analysis
print("Target Variable Distribution:")
target_counts = train_data['Species'].value_counts()
print(target_counts)
print(f"\nClass Balance: {target_counts.std():.2f} (lower is more balanced)")

# Create target distribution plot
fig = px.bar(x=target_counts.index, y=target_counts.values)

# Apply consistent styling
fig.update_traces(marker=dict(color=app_color_palette[:len(target_counts)]))
fig.update_layout(
    height=600,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B5CF6', size=12),
    title_font=dict(color='#7C3AED', size=16),
    xaxis=dict(
        gridcolor='rgba(139,92,246,0.2)',
        zerolinecolor='rgba(139,92,246,0.3)',
        tickfont=dict(color='#8B5CF6', size=11),
        title_font=dict(color='#7C3AED', size=12),
        title="Species"
    ),
    yaxis=dict(
        gridcolor='rgba(139,92,246,0.2)',
        zerolinecolor='rgba(139,92,246,0.3)', 
        tickfont=dict(color='#8B5CF6', size=11),
        title_font=dict(color='#7C3AED', size=12),
        title="Count"
    ),
    legend=dict(font=dict(color='#8B5CF6', size=11))
)

fig.write_html("/Users/yuvalheffetz/ds-agent-projects/session_3e0a615d-9170-492c-b8ab-7121cbd38bd2/research/plots/target_distribution.html", 
               include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

# Feature distribution analysis
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Create feature distribution subplots
fig = make_subplots(rows=2, cols=2, 
                    subplot_titles=feature_columns,
                    vertical_spacing=0.1,
                    horizontal_spacing=0.1)

for i, feature in enumerate(feature_columns):
    row = i // 2 + 1
    col = i % 2 + 1
    
    fig.add_trace(
        go.Histogram(x=train_data[feature], 
                    name=feature,
                    marker_color=app_color_palette[i],
                    nbinsx=15),
        row=row, col=col
    )

fig.update_layout(
    height=600,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B5CF6', size=12),
    showlegend=False
)

# Update all subplot axes with consistent styling
for i in range(1, 5):
    fig.update_xaxes(
        gridcolor='rgba(139,92,246,0.2)',
        zerolinecolor='rgba(139,92,246,0.3)',
        tickfont=dict(color='#8B5CF6', size=10),
        title_font=dict(color='#7C3AED', size=11),
        row=(i-1)//2+1, col=(i-1)%2+1
    )
    fig.update_yaxes(
        gridcolor='rgba(139,92,246,0.2)',
        zerolinecolor='rgba(139,92,246,0.3)',
        tickfont=dict(color='#8B5CF6', size=10),
        title_font=dict(color='#7C3AED', size=11),
        row=(i-1)//2+1, col=(i-1)%2+1
    )

fig.write_html("/Users/yuvalheffetz/ds-agent-projects/session_3e0a615d-9170-492c-b8ab-7121cbd38bd2/research/plots/feature_distributions.html",
               include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

# Outlier detection using IQR method
print("Outlier Analysis using IQR method:")
for feature in feature_columns:
    Q1 = train_data[feature].quantile(0.25)
    Q3 = train_data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = train_data[(train_data[feature] < lower_bound) | (train_data[feature] > upper_bound)]
    print(f"{feature}: {len(outliers)} outliers ({len(outliers)/len(train_data)*100:.1f}%)")

# Feature correlation analysis
correlation_matrix = train_data[feature_columns].corr()
print("Feature Correlation Matrix:")
print(correlation_matrix)

# Create correlation heatmap
fig = px.imshow(correlation_matrix, 
                text_auto=True,
                aspect="auto",
                color_continuous_scale=['#636EFA', '#FFFFFF', '#EF553B'])

fig.update_layout(
    height=600,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B5CF6', size=12),
    title_font=dict(color='#7C3AED', size=16),
    xaxis=dict(
        tickfont=dict(color='#8B5CF6', size=11),
        title_font=dict(color='#7C3AED', size=12)
    ),
    yaxis=dict(
        tickfont=dict(color='#8B5CF6', size=11),
        title_font=dict(color='#7C3AED', size=12)
    ),
    coloraxis_colorbar=dict(
        tickfont=dict(color='#8B5CF6', size=10),
        title_font=dict(color='#7C3AED', size=11)
    )
)

fig.write_html("/Users/yuvalheffetz/ds-agent-projects/session_3e0a615d-9170-492c-b8ab-7121cbd38bd2/research/plots/feature_correlations.html",
               include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

# Class separation analysis using box plots
fig = make_subplots(rows=2, cols=2, 
                    subplot_titles=feature_columns,
                    vertical_spacing=0.1,
                    horizontal_spacing=0.1)

species_list = train_data['Species'].unique()
colors = app_color_palette[:len(species_list)]

for i, feature in enumerate(feature_columns):
    row = i // 2 + 1
    col = i % 2 + 1
    
    for j, species in enumerate(species_list):
        species_data = train_data[train_data['Species'] == species][feature]
        fig.add_trace(
            go.Box(y=species_data,
                   name=species,
                   marker_color=colors[j],
                   showlegend=(i == 0)),  # Only show legend for first subplot
            row=row, col=col
        )

fig.update_layout(
    height=600,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B5CF6', size=12),
    legend=dict(font=dict(color='#8B5CF6', size=11))
)

# Update all subplot axes with consistent styling
for i in range(1, 5):
    fig.update_xaxes(
        gridcolor='rgba(139,92,246,0.2)',
        zerolinecolor='rgba(139,92,246,0.3)',
        tickfont=dict(color='#8B5CF6', size=10),
        title_font=dict(color='#7C3AED', size=11),
        row=(i-1)//2+1, col=(i-1)%2+1
    )
    fig.update_yaxes(
        gridcolor='rgba(139,92,246,0.2)',
        zerolinecolor='rgba(139,92,246,0.3)',
        tickfont=dict(color='#8B5CF6', size=10),
        title_font=dict(color='#7C3AED', size=11),
        row=(i-1)//2+1, col=(i-1)%2+1
    )

fig.write_html("/Users/yuvalheffetz/ds-agent-projects/session_3e0a615d-9170-492c-b8ab-7121cbd38bd2/research/plots/class_separation_boxplots.html",
               include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

# Statistical analysis of class separation
print("Class Separation Analysis:")
for feature in feature_columns:
    print(f"\n{feature}:")
    for species in species_list:
        species_data = train_data[train_data['Species'] == species][feature]
        print(f"  {species}: Mean={species_data.mean():.2f}, Std={species_data.std():.2f}")
        
    # ANOVA test for significant differences between groups
    groups = [train_data[train_data['Species'] == species][feature] for species in species_list]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"  ANOVA: F={f_stat:.2f}, p-value={p_value:.2e} ({'Significant' if p_value < 0.05 else 'Not significant'})")

# Pairwise feature relationships - focus on most discriminative pairs
# Petal scatter plot
fig = px.scatter(train_data, 
                x='PetalLengthCm', 
                y='PetalWidthCm', 
                color='Species',
                color_discrete_sequence=app_color_palette)

fig.update_layout(
    height=600,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B5CF6', size=12),
    title_font=dict(color='#7C3AED', size=16),
    xaxis=dict(
        gridcolor='rgba(139,92,246,0.2)',
        zerolinecolor='rgba(139,92,246,0.3)',
        tickfont=dict(color='#8B5CF6', size=11),
        title_font=dict(color='#7C3AED', size=12),
        title="Petal Length (cm)"
    ),
    yaxis=dict(
        gridcolor='rgba(139,92,246,0.2)',
        zerolinecolor='rgba(139,92,246,0.3)', 
        tickfont=dict(color='#8B5CF6', size=11),
        title_font=dict(color='#7C3AED', size=12),
        title="Petal Width (cm)"
    ),
    legend=dict(font=dict(color='#8B5CF6', size=11), title=dict(text="Species", font=dict(color='#7C3AED')))
)

fig.write_html("/Users/yuvalheffetz/ds-agent-projects/session_3e0a615d-9170-492c-b8ab-7121cbd38bd2/research/plots/petal_scatter_plot.html",
               include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

# Sepal scatter plot for comparison
fig2 = px.scatter(train_data, 
                 x='SepalLengthCm', 
                 y='SepalWidthCm', 
                 color='Species',
                 color_discrete_sequence=app_color_palette)

fig2.update_layout(
    height=600,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B5CF6', size=12),
    title_font=dict(color='#7C3AED', size=16),
    xaxis=dict(
        gridcolor='rgba(139,92,246,0.2)',
        zerolinecolor='rgba(139,92,246,0.3)',
        tickfont=dict(color='#8B5CF6', size=11),
        title_font=dict(color='#7C3AED', size=12),
        title="Sepal Length (cm)"
    ),
    yaxis=dict(
        gridcolor='rgba(139,92,246,0.2)',
        zerolinecolor='rgba(139,92,246,0.3)', 
        tickfont=dict(color='#8B5CF6', size=11),
        title_font=dict(color='#7C3AED', size=12),
        title="Sepal Width (cm)"
    ),
    legend=dict(font=dict(color='#8B5CF6', size=11), title=dict(text="Species", font=dict(color='#7C3AED')))
)

fig2.write_html("/Users/yuvalheffetz/ds-agent-projects/session_3e0a615d-9170-492c-b8ab-7121cbd38bd2/research/plots/sepal_scatter_plot.html",
                include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

print("All plots have been generated successfully!")